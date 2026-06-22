"""
ANPR — Automatic Number Plate Recognition
Pipeline:
  1. Extract frames from video (every N frames)
  2. Per frame: contour-based region detection + direct OCR
  3. Multiple preprocessing variants for different conditions
  4. Regex-validate plate text, deduplicate by highest confidence
  5. Cross-check against authorized-vehicles table
"""
import cv2
import numpy as np
import re
import json
import threading
from pathlib import Path

_reader = None
_reader_lock = threading.Lock()


def _get_reader():
    global _reader
    with _reader_lock:
        if _reader is None:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess_variants(img_bgr):
    """Return several preprocessed versions of a BGR image for maximum OCR coverage."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variants = [img_bgr, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

    # Adaptive threshold
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    variants.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))

    # Inverted adaptive threshold (white-on-dark plates)
    inv = cv2.bitwise_not(thr)
    variants.append(cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR))

    # Bilateral filter + Otsu
    bil = cv2.bilateralFilter(gray, 9, 75, 75)
    _, otsu = cv2.threshold(bil, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    return variants


# ---------------------------------------------------------------------------
# Plate region detection via contours
# ---------------------------------------------------------------------------

def _find_plate_regions(frame):
    """Return (x1,y1,x2,y2) bounding boxes of candidate plate regions."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny + dilate to link text components on plate
    edges = cv2.Canny(blur, 50, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 4))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0:
            continue
        ratio = cw / ch
        area = cw * ch
        # Typical plate: width 2–7× the height, minimum pixel area
        if 1.5 < ratio < 8.0 and area > 1200 and cw > 50 and ch > 12:
            px = max(int(cw * 0.06), 4)
            py = max(int(ch * 0.15), 3)
            x1 = max(0, x - px)
            y1 = max(0, y - py)
            x2 = min(w, x + cw + px)
            y2 = min(h, y + ch + py)
            regions.append((x1, y1, x2, y2))

    return regions


# ---------------------------------------------------------------------------
# Plate text validation
# ---------------------------------------------------------------------------

# Indian plates: 2 letters + 2 digits + 1–3 letters + 4 digits, any spacing
_PLATE_RE = re.compile(
    r"\b([A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{4})\b",
    re.IGNORECASE,
)
# Generic fallback: 5–12 alphanumeric chars with both letters and digits
_GENERIC_RE = re.compile(r"\b[A-Z0-9]{5,12}\b", re.IGNORECASE)

def _clean(text: str) -> str:
    return re.sub(r"[\s\-\.]", "", text.upper())


# Indian-plate-aware correction:
# Format: [A-Z]{2} [digit]{2} [A-Z]{1,3} [digit]{4}
# Positions 3-4 must be digits → fix O→0, I/l→1, S→5, B→8, G→6, Z→2
# Positions 1-2 and 5-7 must be letters → fix 0→O, 1→I, 8→B, 5→S, 6→G
_IP_PATTERN = re.compile(
    r"([A-Z0-9]{2})([A-Z0-9]{2})([A-Z0-9]{1,3})([A-Z0-9]{4})",
    re.IGNORECASE,
)

_DIGIT_FIXES = str.maketrans("OILSBGZ", "0115862")   # common letter→digit errors
_ALPHA_FIXES = str.maketrans("01589",  "OISBG")       # common digit→letter errors


def _fix_indian_plate(m: re.Match) -> str:
    state    = m.group(1).upper().translate(_ALPHA_FIXES)   # 2 letters
    district = m.group(2).upper().translate(_DIGIT_FIXES)   # 2 digits
    series   = m.group(3).upper().translate(_ALPHA_FIXES)   # 1-3 letters
    number   = m.group(4).upper().translate(_DIGIT_FIXES)   # 4 digits
    return state + district + series + number


def _apply_ocr_fixes(text: str) -> str:
    """Apply position-aware character correction for Indian plate format."""
    # First pass: run the structured Indian-plate fix
    corrected = _IP_PATTERN.sub(_fix_indian_plate, text.upper())
    return corrected


def _is_valid_plate(text: str) -> bool:
    c = _clean(text)
    has_alpha = bool(re.search(r"[A-Z]", c))
    has_digit = bool(re.search(r"\d", c))
    return has_alpha and has_digit and 5 <= len(c) <= 12


def _extract_plates_from_text(text: str, conf: float) -> list[tuple[str, float]]:
    """Try to find valid plate strings within a raw OCR text result."""
    found = []

    # Try Indian format first (most specific)
    for m in _PLATE_RE.finditer(text):
        plate = _clean(_apply_ocr_fixes(m.group(0)))
        if _is_valid_plate(plate):
            found.append((plate, conf))

    # Generic fallback
    if not found:
        for m in _GENERIC_RE.finditer(text):
            plate = _clean(_apply_ocr_fixes(m.group(0)))
            if _is_valid_plate(plate):
                found.append((plate, conf * 0.85))  # slight penalty for generic match

    return found


# ---------------------------------------------------------------------------
# OCR on a single image region
# ---------------------------------------------------------------------------

def _ocr_region(img_bgr, reader, min_conf=0.25) -> dict[str, float]:
    """Run OCR on one region across all preprocessing variants; return plate→best_conf."""
    h, w = img_bgr.shape[:2]
    # Upscale if too small for OCR
    if w < 200:
        scale = 220 / w
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    # Cap at reasonable width
    if w > 900:
        scale = 900 / w
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    plates: dict[str, float] = {}
    for variant in _preprocess_variants(img_bgr):
        try:
            results = reader.readtext(variant, detail=1, paragraph=False)
            for (_, text, conf) in results:
                if conf < min_conf:
                    continue
                for plate, c in _extract_plates_from_text(text, conf):
                    if plate not in plates or plates[plate] < c:
                        plates[plate] = c
        except Exception:
            pass
    return plates


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def _process_frame(frame, reader) -> dict[str, float]:
    h, w = frame.shape[:2]

    # Downscale very large frames for speed
    if w > 1280:
        scale = 1280 / w
        frame_small = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        frame_small = frame

    all_plates: dict[str, float] = {}

    # --- Method 1: contour-based plate regions ---
    for (x1, y1, x2, y2) in _find_plate_regions(frame_small):
        roi = frame_small[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        for plate, conf in _ocr_region(roi, reader).items():
            if plate not in all_plates or all_plates[plate] < conf:
                all_plates[plate] = conf

    # --- Method 2: direct OCR on bottom 60 % of frame (plates usually there) ---
    bottom = frame_small[int(frame_small.shape[0] * 0.35):]
    try:
        results = reader.readtext(bottom, detail=1, paragraph=False, min_size=10)
        for (_, text, conf) in results:
            if conf < 0.30:
                continue
            for plate, c in _extract_plates_from_text(text, conf):
                if plate not in all_plates or all_plates[plate] < c:
                    all_plates[plate] = c
    except Exception:
        pass

    return all_plates


# ---------------------------------------------------------------------------
# Full video processing (called in background thread)
# ---------------------------------------------------------------------------

def process_video_job(job_id: str, video_path: Path, frame_skip: int = 12):
    """
    Process every `frame_skip`-th frame of the video.
    Writes progress to anpr_jobs table in real time.
    """
    from app.database import get_conn

    conn = get_conn()
    c = conn.cursor()

    def _update(status, processed, total, plates_json="[]", error=""):
        c.execute(
            "UPDATE anpr_jobs SET status=?, processed_frames=?, total_frames=?, plates_found=?, error=? WHERE id=?",
            (status, processed, total, plates_json, error, job_id),
        )
        conn.commit()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _update("error", 0, 0, "[]", "Cannot open video file")
        conn.close()
        return

    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    _update("processing", 0, total)

    _update("processing", 0, total)  # show "initializing EasyOCR"
    try:
        reader = _get_reader()
    except Exception as e:
        _update("error", 0, total, "[]", f"EasyOCR init failed: {e}")
        cap.release()
        conn.close()
        return

    all_plates: dict[str, float] = {}  # plate -> best confidence
    frame_num = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_skip == 0:
            detected = _process_frame(frame, reader)
            for plate, conf in detected.items():
                if plate not in all_plates or all_plates[plate] < conf:
                    all_plates[plate] = conf

            processed_count += 1
            # Update DB every 5 processed frames
            if processed_count % 5 == 0:
                _update("processing", frame_num, total)

        frame_num += 1

    cap.release()

    def _norm(s):
        """Collapse visually-similar chars so OCR variants match the whitelist."""
        return (
            s.upper()
            .replace(" ", "").replace("-", "")
            .replace("O", "0")
            .replace("I", "1").replace("L", "1")
            .replace("S", "5")
            .replace("B", "8")
            .replace("G", "6")
            .replace("Z", "2")
            .replace("Q", "0")
        )

    def _edit_distance(a, b):
        """Simple Levenshtein distance."""
        if a == b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
            prev = curr
        return prev[-1]

    # Build final results with authorization check
    all_rows = c.execute("SELECT * FROM authorized_vehicles").fetchall()
    results = []
    for plate, conf in sorted(all_plates.items(), key=lambda x: -x[1]):
        norm_plate = _norm(plate)
        # 1) exact normalized match
        row = next((r for r in all_rows if _norm(r["plate"]) == norm_plate), None)
        # 2) fuzzy fallback: edit-distance ≤ 1 on normalized strings (catches 3↔S, A↔4 etc.)
        if row is None:
            closest = min(all_rows, key=lambda r: _edit_distance(_norm(r["plate"]), norm_plate))
            if _edit_distance(_norm(closest["plate"]), norm_plate) <= 1:
                row = closest

        authorized = row is not None
        owner = row["owner"] if row else "Unknown"
        vehicle_type = row["vehicle_type"] if row else "Unknown"
        decision = "GRANTED" if authorized else "DENIED"

        c.execute(
            "INSERT INTO access_log (plate, confidence, authorized, decision, source) VALUES (?,?,?,?,'ANPR')",
            (plate, round(conf, 4), 1 if authorized else 0, decision),
        )

        results.append({
            "plate": plate,
            "confidence": round(conf, 3),
            "authorized": authorized,
            "owner": owner,
            "vehicle_type": vehicle_type,
            "decision": decision,
        })

    plates_json = json.dumps(results)
    c.execute(
        "UPDATE anpr_jobs SET status='completed', processed_frames=?, plates_found=?, completed_at=datetime('now') WHERE id=?",
        (frame_num, plates_json, job_id),
    )
    conn.commit()
    conn.close()
