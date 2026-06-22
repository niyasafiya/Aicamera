"""
License-plate OCR — modelled on YardMonitor/core/ocr.py.

Strategy (mirrors YardMonitor):
  - Upscale plate crop to a minimum readable size
  - Run EasyOCR over FIVE preprocessing variants (base + 4 enhanced)
  - Confidence-weighted character majority vote to pick the final plate text
  - Position-aware character correction for Indian plate format

For full-frame ANPR (demo video):
  - Detect vehicle bounding boxes via YOLO (caller's responsibility)
  - Extract plate region with contour finder
  - Run multi-pass OCR on the plate crop
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# EasyOCR (lazy init, shared reader)
# ---------------------------------------------------------------------------
_reader = None
MIN_OCR_CONF  = 0.35
MIN_VAR_CONF  = 0.20   # lower threshold for variant passes (like YardMonitor)
MIN_PLATE_CHARS = 4


def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            print("[EasyOCR] Initialising… (first call, may take ~10 s)")
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("[EasyOCR] Ready.")
        except Exception as exc:
            print(f"[EasyOCR] Init failed: {exc}")
            _reader = "FAILED"
    return None if _reader == "FAILED" else _reader


# ---------------------------------------------------------------------------
# Text cleaning & plate validation
# ---------------------------------------------------------------------------
_PLATE_RE = re.compile(r"^[A-Z0-9]{4,13}$")

# Indian plate format: SS DD L(LL) NNNN
# e.g.  KL07CK4521  TN38BA1190  MH12AB9012
_INDIAN_RE = re.compile(
    r"^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$"
)

# Common OCR confusions — position-aware (like YardMonitor's correction table)
_ALPHA_TO_DIGIT = str.maketrans("OIZSB", "01258")
_DIGIT_TO_ALPHA = str.maketrans("01258", "OIZSB")


def _clean(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _looks_like_plate(t: str) -> bool:
    if not _PLATE_RE.match(t):
        return False
    return (sum(c.isalpha() for c in t) >= 1 and
            sum(c.isdigit() for c in t) >= 1)


def _correct_indian(text: str) -> str:
    """
    Apply position-aware character correction for Indian plates.
    Positions 0-1: letters (state code) — fix digit↔alpha confusion.
    Positions 2-3: digits (district code) — fix alpha↔digit confusion.
    Positions 4-6: letters (series).
    Positions 7-10: digits (number).
    """
    m = _INDIAN_RE.match(text)
    if not m:
        return text
    state  = m.group(1).translate(_DIGIT_TO_ALPHA)
    dist   = m.group(2).translate(_ALPHA_TO_DIGIT)
    series = m.group(3).translate(_DIGIT_TO_ALPHA)
    num    = m.group(4).translate(_ALPHA_TO_DIGIT)
    return f"{state}{dist}{series}{num}"


# ---------------------------------------------------------------------------
# Image preprocessing variants  (mirrors YardMonitor's five passes)
# ---------------------------------------------------------------------------

def _upscale(img: np.ndarray, min_h: int = 60, min_w: int = 200) -> np.ndarray:
    """Upscale so EasyOCR sees enough pixels."""
    h, w = img.shape[:2]
    scale = max(min_h / max(h, 1), min_w / max(w, 1), 1.0)
    if scale > 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    return img


def _with_border(img: np.ndarray, pad: int = 10) -> np.ndarray:
    """White border prevents EasyOCR from clipping edge characters."""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _var_a(gray: np.ndarray) -> np.ndarray:
    """CLAHE + unsharp mask — good for low-contrast plates."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharp    = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _var_b(gray: np.ndarray) -> np.ndarray:
    """Bilateral filter + Otsu — dark text on bright plate."""
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _var_c(gray: np.ndarray) -> np.ndarray:
    """Inverted Otsu — bright text on dark plate."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _var_d(gray: np.ndarray) -> np.ndarray:
    """Morphological closing + adaptive threshold — handles dirt/damage."""
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed  = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    thresh  = cv2.adaptiveThreshold(
        closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _make_variants(crop: np.ndarray) -> List[np.ndarray]:
    """Return [base, var_a, var_b, var_c, var_d] — all ready for EasyOCR."""
    up   = _upscale(crop)
    base = _with_border(up)
    gray = _to_gray(up)
    return [
        base,
        _with_border(_var_a(gray)),
        _with_border(_var_b(gray)),
        _with_border(_var_c(gray)),
        _with_border(_var_d(gray)),
    ]


# ---------------------------------------------------------------------------
# Majority-vote OCR (mirrors YardMonitor's multi-pass strategy)
# ---------------------------------------------------------------------------

_PLATE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def _ocr_image(reader, img: np.ndarray, min_conf: float) -> List[Tuple[str, float]]:
    """Run EasyOCR on one image, return [(text, conf)] for plate-like hits."""
    try:
        results = reader.readtext(
            img, detail=1, paragraph=False,
            decoder='greedy',
            allowlist=_PLATE_CHARS,
            canvas_size=480,   # CRAFT detection network max resolution — smaller = faster
            mag_ratio=1.0,     # no internal magnification
            min_size=10,       # skip tiny text regions
            width_ths=0.7,     # merge nearby text boxes faster
        )
    except Exception:
        return []
    out = []
    for (_bbox, text, conf) in results:
        if conf < min_conf:
            continue
        t = _clean(text)
        if _looks_like_plate(t) and len(t) >= MIN_PLATE_CHARS:
            out.append((t, conf))
    return out


def _majority_vote(candidates: List[Tuple[str, float]]) -> Optional[Tuple[str, float]]:
    """
    Character-level majority vote weighted by confidence (YardMonitor style).
    Groups candidates by length first (most common length wins),
    then picks the highest-confidence character at each position.
    """
    if not candidates:
        return None

    # Pick most common length
    from collections import Counter
    lens   = Counter(len(t) for t, _ in candidates)
    best_len = lens.most_common(1)[0][0]
    filtered = [(t, c) for t, c in candidates if len(t) == best_len]
    if not filtered:
        filtered = candidates

    # Char-level vote
    result_chars = []
    for pos in range(best_len):
        char_votes: Dict[str, float] = {}
        for text, conf in filtered:
            if pos < len(text):
                ch = text[pos]
                char_votes[ch] = char_votes.get(ch, 0.0) + conf
        if char_votes:
            result_chars.append(max(char_votes, key=char_votes.__getitem__))

    plate = "".join(result_chars)
    avg_conf = sum(c for _, c in filtered) / len(filtered)
    return plate, round(avg_conf, 3)


def read_plate_crop(crop: np.ndarray) -> Optional[Dict]:
    """
    Multi-pass OCR on a pre-cropped plate image.
    Returns {'plate': str, 'confidence': float} or None.
    """
    reader = _get_reader()
    if reader is None or crop is None or crop.size == 0:
        return None

    variants  = _make_variants(crop)
    thresholds = [MIN_OCR_CONF, MIN_VAR_CONF, MIN_VAR_CONF, MIN_VAR_CONF, MIN_VAR_CONF]
    candidates: List[Tuple[str, float]] = []

    for img, thr in zip(variants, thresholds):
        candidates.extend(_ocr_image(reader, img, thr))

    result = _majority_vote(candidates)
    if not result:
        return None

    plate, conf = result
    plate = _correct_indian(plate)
    return {"plate": plate, "confidence": conf}


# ---------------------------------------------------------------------------
# Full-frame plate extraction (used by ANPR video job)
# ---------------------------------------------------------------------------

def extract_plates_from_frame(frame: np.ndarray) -> List[Dict]:
    """
    Run multi-pass OCR directly on a full video frame.
    Used when no separate plate-detector model is available.
    """
    reader = _get_reader()
    if reader is None:
        return []

    # Crop lower 65% — plates are never in the sky/ceiling area, halves CRAFT scan area
    h, w = frame.shape[:2]
    frame = frame[int(h * 0.35):, :]

    # Resize to 480 px wide max — still legible for plates
    h, w = frame.shape[:2]
    if w > 480:
        scale = 480 / w
        frame = cv2.resize(frame, (480, int(h * scale)))

    images = [_with_border(frame)]

    candidates: List[Tuple[str, float]] = []
    for img in images:
        candidates.extend(_ocr_image(reader, img, MIN_OCR_CONF))

    # Deduplicate by plate text, keep highest confidence
    best: Dict[str, float] = {}
    for plate, conf in candidates:
        if plate not in best or conf > best[plate]:
            best[plate] = conf

    return [{"plate": _correct_indian(p), "confidence": c} for p, c in best.items()]
