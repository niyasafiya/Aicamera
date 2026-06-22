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

_PLATE_OK   = re.compile(r"[A-Z0-9]")
_PLATE_CORE = re.compile(r"[A-Z]{2}\d{2}[A-Z]{1,3}\d{3,4}")

MIN_PLATE_CHARS  = 4
_VARIANT_MIN_CONF = 0.25
_OCR_MIN_CONF     = 0.35

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
        from paddleocr import PaddleOCR
        _paddle = PaddleOCR(use_angle_cls=False, lang="en",
                            use_gpu=False, show_log=False)
        _ocr_backend = "paddle"
        log.info("[ANPR] PaddleOCR ready")
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

def _upscale(img: np.ndarray, min_h: int = 200, min_w: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(min_h / max(h, 1), min_w / max(w, 1), 1.0)
    if scale > 1.0:
        new_w = min(int(w * scale), 2000)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img


def _add_border(img: np.ndarray, px: int = 15) -> np.ndarray:
    val = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.copyMakeBorder(img, px, px, px, px,
                              cv2.BORDER_CONSTANT, value=val)


def _preprocess_variants(img: np.ndarray) -> List[np.ndarray]:
    """Four preprocessed variants (A=CLAHE+unsharp, B=bilateral+Otsu,
    C=inverted Otsu, D=morph+adaptive) plus gamma correction."""
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

    # B — bilateral + Otsu
    vb = cv2.bilateralFilter(gray, 11, 17, 17)
    _, vb = cv2.threshold(vb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # C — inverted Otsu (white-on-dark / reflective plates)
    _, vc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # D — morph close + adaptive threshold (fills dirt/wear breaks)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    vd = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    vd = cv2.adaptiveThreshold(vd, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, blockSize=11, C=5)

    return [cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for v in [va, vb, vc, vd]]


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
    """Position-aware letter/digit fix for Indian format SS DD (L|LL) NNNN."""
    # Extract plate pattern from longer strings (bumper sticker noise)
    if len(plate) > 10:
        m = _PLATE_CORE.search(plate)
        if m:
            plate = m.group(0)

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

    # Pass 1: base image, det=False
    candidates.extend(_paddle_ocr_once(paddle, base, det=False))

    # Passes 2-5: preprocessed variants, det=False
    for variant in _preprocess_variants(base):
        candidates.extend(_paddle_ocr_once(paddle, variant, det=False))

    # Pass 6: base, det=True (EAST/DB detector finds char bboxes)
    candidates.extend(_paddle_ocr_once(paddle, base, det=True))

    result = _majority_vote(candidates)
    if not result:
        return None
    plate, conf = result
    plate = _postfix(plate)
    if not _looks_like_plate(plate):
        return None
    return plate, conf


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

    for img in [base] + _preprocess_variants(base):
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

def _find_plate_regions(frame: np.ndarray) -> List[tuple]:
    """Return (x1,y1,x2,y2) boxes of candidate plate regions via contours."""
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

    regions = []
    min_area = w * h * 0.008
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < min_area:
            continue
        aspect = cw / max(ch, 1)
        if not (2.0 <= aspect <= 7.0):
            continue
        pad = 5
        regions.append((
            max(0, x - pad), max(0, y - pad),
            min(w, x + cw + pad), min(h, y + ch + pad),
        ))
    return regions


# ---------------------------------------------------------------------------
# Public: extract plates from a full video frame
# ---------------------------------------------------------------------------

def extract_plates_from_frame(frame: np.ndarray) -> List[Dict]:
    """Run plate OCR on a full video frame.
    Uses contour detection to find plate regions first (fast), then
    runs PaddleOCR/EasyOCR only on those small crops."""

    # Work on the bottom 50% — plates are never in sky/ceiling
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.50):, :]

    # Downscale for contour detection (pure OpenCV — no neural net cost)
    rh, rw = roi.shape[:2]
    if rw > 640:
        scale = 640 / rw
        roi_small = cv2.resize(roi, (640, int(rh * scale)))
    else:
        roi_small = roi
        scale = 1.0

    best: Dict[str, float] = {}

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
                if p not in best or c > best[p]:
                    best[p] = c

    if not best:
        # Fallback: OCR on the entire bottom strip at reduced size
        if rw > 320:
            fscale = 320 / rw
            roi_fb = cv2.resize(roi, (320, int(rh * fscale)))
        else:
            roi_fb = roi
        result = read_plate_crop(roi_fb)
        if result:
            best[result["plate"]] = result["confidence"]

    return [{"plate": p, "confidence": round(c, 3)} for p, c in best.items()]
