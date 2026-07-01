"""
License-plate OCR — PaddleOCR (PP-OCRv4) backend.
Ported from YardMonitor/core/ocr.py.

Why PaddleOCR over EasyOCR:
  - ~4-6x faster per crop (PP-OCRv4 mobile vs CRAFT+CRNN)
  - Better accuracy on digit/letter confusion (1/I, 0/O, S/5, B/8, etc.)
  - No character-split artefacts (R→Y+4 was an EasyOCR segmentation bug)
  - Models auto-download to ~/.paddleocr/ on first run (~50 MB)

Falls back to EasyOCR automatically if PaddleOCR is not installed.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character correction tables  (from YardMonitor)
# ---------------------------------------------------------------------------

# Digit → Letter  (state code and series positions must be letters)
_D2L = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S",
                       "6": "G", "7": "Z", "8": "B"})
# Letter → Digit  (district and serial positions must be digits)
_L2D = str.maketrans({"B": "8", "D": "0", "G": "6", "I": "1", "J": "1",
                       "L": "1", "O": "0", "Q": "0", "S": "5", "Z": "2"})

_PLATE_OK    = re.compile(r"[A-Z0-9]")
_PLATE_CORE  = re.compile(r"[A-Z]{2}\d{2}[A-Z]{1,3}\d{3,4}")
# Dubai plate: 1-3 letters then 1-5 digits  e.g. A12345, AB1234, XA99999
_DUBAI_PLATE = re.compile(r"^[A-Z]{1,3}\d{1,5}$")

MIN_PLATE_CHARS  = 3
_VARIANT_MIN_CONF = 0.10
_OCR_MIN_CONF     = 0.18

# ---------------------------------------------------------------------------
# Lazy OCR reader (PaddleOCR with EasyOCR fallback)
# ---------------------------------------------------------------------------

_paddle = None
_easyocr_reader = None
_ocr_backend = None   # "paddle" | "easyocr" | None


def _get_paddle():
    global _paddle, _ocr_backend
    if _paddle is not None:
        return _paddle
    try:
        import os
        from paddleocr import PaddleOCR
        # Speed: MKL-DNN gives a 2-3x CPU inference speedup with the *same* model
        # and identical accuracy. Cap cpu_threads to leave one core free so the
        # asyncio event loop (health check, job polling, MJPEG streams) stays
        # responsive during a scan — this is what previously made the backend
        # appear to go "offline" mid-scan when OCR saturated every core.
        cpu_threads = max(2, (os.cpu_count() or 4) - 1)
        _paddle = PaddleOCR(use_angle_cls=False, lang="en",
                            use_gpu=False, show_log=False,
                            enable_mkldnn=True, cpu_threads=cpu_threads)
        _ocr_backend = "paddle"
        log.info("[ANPR] PaddleOCR ready (mkldnn, %d threads)", cpu_threads)
        return _paddle
    except Exception as exc:
        log.warning("[ANPR] PaddleOCR unavailable (%s) — using EasyOCR fallback", exc)
        _paddle = "FAILED"
        return None


def _get_easyocr():
    global _easyocr_reader, _ocr_backend
    if _easyocr_reader is not None:
        return None if _easyocr_reader == "FAILED" else _easyocr_reader
    try:
        import easyocr
        log.info("[ANPR] Initialising EasyOCR…")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _ocr_backend = "easyocr"
        log.info("[ANPR] EasyOCR ready")
        return _easyocr_reader
    except Exception as exc:
        log.error("[ANPR] EasyOCR init failed: %s", exc)
        _easyocr_reader = "FAILED"
        return None


# ---------------------------------------------------------------------------
# Preprocessing variants  (mirrors YardMonitor)
# ---------------------------------------------------------------------------

def _upscale(img: np.ndarray, min_h: int = 128, min_w: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(min_h / max(h, 1), min_w / max(w, 1), 1.0)
    if scale > 1.0:
        new_w = min(int(w * scale), 1600)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img


def _add_border(img: np.ndarray, px: int = 15) -> np.ndarray:
    val = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.copyMakeBorder(img, px, px, px, px,
                              cv2.BORDER_CONSTANT, value=val)


def _preprocess_variants(img: np.ndarray) -> List[np.ndarray]:
    """Variant A only: CLAHE + unsharp mask with gamma correction.
    Callers access variants[0]; B/C/D were never used so we skip them."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # Gamma brightness correction
    mean_val = float(np.mean(gray))
    if mean_val < 80:
        gamma = 0.40
    elif mean_val < 110:
        gamma = 0.65
    elif mean_val > 210:
        gamma = 1.8
    else:
        gamma = None
    if gamma is not None:
        lut  = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
        gray = cv2.LUT(gray, lut)
    gray = cv2.medianBlur(gray, 3)

    # A — CLAHE + unsharp mask
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
    va    = clahe.apply(gray)
    blur  = cv2.GaussianBlur(va, (5, 5), 0)
    va    = cv2.addWeighted(va, 1.6, blur, -0.6, 0)

    return [cv2.cvtColor(va, cv2.COLOR_GRAY2BGR)]


# ---------------------------------------------------------------------------
# Majority vote  (from YardMonitor — length-score weighted)
# ---------------------------------------------------------------------------

def _majority_vote(candidates: List[Tuple[str, float]]) -> Optional[Tuple[str, float]]:
    """Confidence-weighted character-level majority vote.
    Prefers the longest plate among lengths whose score is ≥ 60 % of best
    (handles OCR dropping trailing characters)."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    len_stats: Dict[int, List[float]] = {}
    for p, c in candidates:
        len_stats.setdefault(len(p), []).append(c)

    len_scores = {n: len(vs) * float(np.mean(vs)) for n, vs in len_stats.items()}
    best_score = max(len_scores.values())
    target_len = max(
        n for n, s in len_scores.items() if s >= best_score * 0.60
    )
    matching = [(p, c) for p, c in candidates if len(p) == target_len]
    if not matching:
        return max(candidates, key=lambda x: x[1])

    voted = []
    for pos in range(target_len):
        char_scores: Dict[str, float] = {}
        for p, conf in matching:
            ch = p[pos]
            char_scores[ch] = char_scores.get(ch, 0.0) + conf
        voted.append(max(char_scores, key=char_scores.__getitem__))

    plate    = "".join(voted)
    avg_conf = float(np.mean([c for _, c in matching]))
    return plate, round(avg_conf, 3)


# ---------------------------------------------------------------------------
# Postfix correction  (from YardMonitor — position-aware for Indian plates)
# ---------------------------------------------------------------------------

def _postfix(plate: str) -> str:
    """Position-aware letter/digit fix.
    Detects Dubai format (1-3 letters + 1-5 digits) vs Indian format automatically."""
    # Extract plate pattern from longer strings (bumper sticker noise)
    if len(plate) > 10:
        m = _PLATE_CORE.search(plate)
        if m:
            plate = m.group(0)

    # Dubai format: all letters come first, then all digits
    first_digit = next((i for i, ch in enumerate(plate) if ch.isdigit()), len(plate))
    lc, dc = first_digit, len(plate) - first_digit
    if 1 <= lc <= 3 and 1 <= dc <= 5:
        # Minimal fix: ensure letter zone has letters, digit zone has digits
        letters = "".join(
            ch if ch.isalpha() else ch.translate(_D2L) for ch in plate[:lc]
        )
        digits = "".join(
            ch if ch.isdigit() else ch.translate(_L2D) for ch in plate[lc:]
        )
        return letters + digits

    # Indian format: position-aware correction
    n = len(plate)
    if n == 8:
        lp = {0, 1, 4};      dp = {2, 3, 5, 6, 7}
    elif n == 9:
        lp = {0, 1, 4};      dp = {2, 3, 5, 6, 7, 8}
    elif n == 10:
        lp = {0, 1, 4, 5};   dp = {2, 3, 6, 7, 8, 9}
    elif n >= 6:
        lp = {0, 1};         dp = {2, 3} | set(range(max(4, n - 4), n))
    else:
        return plate

    result = list(plate)
    for i, ch in enumerate(plate):
        if i in lp and ch.isdigit():
            result[i] = ch.translate(_D2L)
        elif i in dp and ch.isalpha():
            result[i] = ch.translate(_L2D)
    return "".join(result)


def _looks_like_plate(t: str) -> bool:
    return (len(t) >= MIN_PLATE_CHARS
            and any(c.isalpha() for c in t)
            and any(c.isdigit() for c in t))


# ---------------------------------------------------------------------------
# PaddleOCR pass
# ---------------------------------------------------------------------------

def _paddle_ocr_once(paddle, img: np.ndarray, det: bool) -> List[Tuple[str, float]]:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        result = paddle.ocr(img, det=det, rec=True, cls=False)
    except Exception as exc:
        log.debug("PaddleOCR error: %s", exc)
        return []
    if not result or result[0] is None:
        return []

    out = []
    for row in result[0]:
        try:
            if det:
                text, conf = row[1][0], float(row[1][1])
            else:
                text, conf = row[0], float(row[1])
            if conf < _VARIANT_MIN_CONF:
                continue
            cleaned = "".join(ch for ch in text.upper() if _PLATE_OK.match(ch))
            if cleaned and _looks_like_plate(cleaned):
                out.append((cleaned, conf))
        except (IndexError, TypeError, ValueError):
            continue
    return out


def _ocr_with_paddle(crop: np.ndarray) -> Optional[Tuple[str, float]]:
    paddle = _get_paddle()
    if paddle is None or paddle == "FAILED":
        return None

    base = _add_border(_upscale(crop))
    candidates: List[Tuple[str, float]] = []

    # Pass 1: base image, recognition only
    p1 = _paddle_ocr_once(paddle, base, det=False)
    candidates.extend(p1)
    # Early exit — skip the (expensive) pass-2 variant when pass 1 is already
    # decent. During a video scan, cross-frame majority voting (_cluster_and_vote)
    # corrects residual character confusions across frames, so per-crop pass-2
    # correction is largely redundant. 0.60 (was 0.70) skips pass 2 on more reads.
    if p1 and max(c for _, c in p1) >= 0.60:
        result = _majority_vote(candidates)
        if result:
            plate = _postfix(result[0])
            if _looks_like_plate(plate):
                return plate, round(result[1], 3)

    # Pass 2: CLAHE variant — always run to provide majority-vote error correction.
    # Combining pass 1 + pass 2 catches character confusions (0/O, 1/I, 5/S, B/8).
    variants = _preprocess_variants(base)
    candidates.extend(_paddle_ocr_once(paddle, variants[0], det=False))
    if candidates:
        result = _majority_vote(candidates)
        if result and result[1] >= 0.40:
            plate = _postfix(result[0])
            if _looks_like_plate(plate):
                return plate, round(result[1], 3)

    # Pass 3: det=True — last resort only when both rec passes found nothing
    if not candidates:
        candidates.extend(_paddle_ocr_once(paddle, base, det=True))

    result = _majority_vote(candidates)
    if not result:
        return None
    plate = _postfix(result[0])
    if not _looks_like_plate(plate):
        return None
    return plate, round(result[1], 3)


# ---------------------------------------------------------------------------
# EasyOCR pass (fallback)
# ---------------------------------------------------------------------------

_EASYOCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _ocr_with_easyocr(crop: np.ndarray) -> Optional[Tuple[str, float]]:
    reader = _get_easyocr()
    if reader is None:
        return None

    base = _add_border(_upscale(crop))
    candidates: List[Tuple[str, float]] = []
    variants = _preprocess_variants(base)

    for img in [base, variants[0]]:   # base + CLAHE only
        try:
            results = reader.readtext(
                img, detail=1, paragraph=False,
                decoder="greedy",
                allowlist=_EASYOCR_ALLOWLIST,
                canvas_size=320,
                mag_ratio=1.0,
                min_size=10,
                width_ths=0.7,
            )
            for (_, text, conf) in results:
                if conf < _VARIANT_MIN_CONF:
                    continue
                cleaned = "".join(ch for ch in text.upper() if _PLATE_OK.match(ch))
                if cleaned and _looks_like_plate(cleaned):
                    candidates.append((cleaned, conf))
        except Exception:
            pass

    result = _majority_vote(candidates)
    if not result:
        return None
    plate, conf = result
    plate = _postfix(plate)
    if not _looks_like_plate(plate):
        return None
    return plate, conf


# ---------------------------------------------------------------------------
# Public: read a pre-cropped plate image
# ---------------------------------------------------------------------------

def read_plate_crop(crop: np.ndarray) -> Optional[Dict]:
    """Multi-pass OCR on a pre-cropped plate image.
    Returns {'plate': str, 'confidence': float} or None."""
    if crop is None or crop.size == 0:
        return None

    # Try PaddleOCR first, fall back to EasyOCR
    result = _ocr_with_paddle(crop)
    if result is None:
        result = _ocr_with_easyocr(crop)
    if result is None:
        return None

    plate, conf = result
    return {"plate": plate, "confidence": round(conf, 3)}


# ---------------------------------------------------------------------------
# Contour-based plate region finder  (from YardMonitor's _find_plate_contour)
# ---------------------------------------------------------------------------

def _find_plate_regions(frame: np.ndarray, max_regions: int = 3) -> List[tuple]:
    """Return (x1,y1,x2,y2) boxes of candidate plate regions via contours,
    ranked by plate-likeness and capped to the best `max_regions`. Capping is the
    main speed lever: OCR runs only on a few strong candidates per frame instead
    of on every blob (cluttered scenes can otherwise produce 10+ false regions,
    each costing a full multi-pass OCR)."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    # CLAHE to boost contrast on dirty/low-light plates
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 40, 120)

    # Horizontal morphological close merges character edges into one plate blob
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 5))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scored = []
    min_area = w * h * 0.004
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area:
            continue
        aspect = cw / max(ch, 1)
        # 1.2 lower-bound catches Dubai two-row plates; 9.0 upper-bound for wide strips
        if not (1.2 <= aspect <= 9.0):
            continue
        # Rank: larger blobs whose aspect is near a typical plate (~3.3) score highest.
        score = area * (1.0 / (1.0 + abs(aspect - 3.3)))
        pad = 5
        scored.append((score, (
            max(0, x - pad), max(0, y - pad),
            min(w, x + cw + pad), min(h, y + ch + pad),
        )))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [box for _, box in scored[:max_regions]]


# ---------------------------------------------------------------------------
# Public: extract plates from a full video frame
# ---------------------------------------------------------------------------

def quick_has_plate(frame: np.ndarray) -> bool:
    """Fast contour-only check: does this frame have any plate-like regions?
    No OCR — used to pre-filter frames before expensive OCR pass."""
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.45):, :]
    rh, rw = roi.shape[:2]
    if rw > 640:
        roi_small = cv2.resize(roi, (640, int(rh * 640 / rw)))
    else:
        roi_small = roi
    return len(_find_plate_regions(roi_small)) > 0


def extract_plates_from_frame(frame: np.ndarray) -> List[Dict]:
    """Run plate OCR on a full video frame.
    Uses contour detection to find plate regions first (fast), then
    runs PaddleOCR/EasyOCR only on those small crops.
    Returns list of dicts with keys: plate, confidence, bbox (full-frame coords or None)."""

    # Work on the bottom 45% — plates are never in sky/ceiling
    h, w = frame.shape[:2]
    roi_y = int(h * 0.45)
    roi = frame[roi_y:, :]

    # Downscale for contour detection (pure OpenCV — no neural net cost)
    rh, rw = roi.shape[:2]
    if rw > 640:
        scale = 640 / rw
        roi_small = cv2.resize(roi, (640, int(rh * scale)))
    else:
        roi_small = roi
        scale = 1.0

    # best[raw_plate] = {"confidence": float, "bbox": tuple|None}
    best: Dict[str, dict] = {}

    regions = _find_plate_regions(roi_small)
    if regions:
        for (x1, y1, x2, y2) in regions:
            # Map coords back to full-size roi
            sx1 = int(x1 / scale); sy1 = int(y1 / scale)
            sx2 = int(x2 / scale); sy2 = int(y2 / scale)
            crop = roi[sy1:sy2, sx1:sx2]
            if crop.size == 0:
                continue
            result = read_plate_crop(crop)
            if result:
                p, c = result["plate"], result["confidence"]
                full_bbox = (sx1, roi_y + sy1, sx2, roi_y + sy2)
                if p not in best or c > best[p]["confidence"]:
                    best[p] = {"confidence": c, "bbox": full_bbox}

    if not best and regions:
        # Fallback: OCR on the entire bottom strip at reduced size.
        # Only when contour detection DID find plate-like structure but the
        # per-region OCR came up empty — a whole-strip retry can still recover it.
        # When no regions were found at all (car not in view) we skip this: a full
        # multi-pass OCR on a structureless strip almost never reads anything and
        # is the biggest per-frame waste during a video scan. The router's
        # end-of-scan fallback still covers clips where contours never fire.
        if rw > 320:
            fscale = 320 / rw
            roi_fb = cv2.resize(roi, (320, int(rh * fscale)))
        else:
            roi_fb = roi
        result = read_plate_crop(roi_fb)
        if result:
            p, c = result["plate"], result["confidence"]
            # Try full-res contour search to get a real bbox for the overlay
            full_regions = _find_plate_regions(roi)
            fb_bbox = None
            if full_regions:
                # Largest region by area is most likely the plate
                bx1, by1, bx2, by2 = max(
                    full_regions, key=lambda r: (r[2] - r[0]) * (r[3] - r[1])
                )
                fb_bbox = (bx1, roi_y + by1, bx2, roi_y + by2)
            best[p] = {"confidence": c, "bbox": fb_bbox}

    return [
        {"plate": p, "confidence": round(v["confidence"], 3), "bbox": v["bbox"]}
        for p, v in best.items()
    ]
