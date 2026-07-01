"""
Face encoding and verification.

Primary engine is DeepFace/Facenet (128-d embedding). When DeepFace/TensorFlow
is unavailable (broken install, offline, etc.) we fall back to a lightweight
OpenCV face-crop embedding so the feature still works end-to-end: registering a
face always stores an encoding, and verifying the same/similar face matches.

Encodings are stored engine-tagged so a query is only ever compared against
stored encodings produced by the *same* engine.

Photos stored under data/faces/, encodings in data/encodings.json.
"""
from __future__ import annotations

import json
import os
import tempfile
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional

FACES_DIR      = Path("data/faces")
ENCODINGS_FILE = Path("data/encodings.json")

# Cosine-similarity thresholds, per engine.
THRESHOLDS = {
    "facenet": 0.72,   # DeepFace/Facenet 128-d
    "opencv":  0.80,   # raw face-crop fallback
}
DEFAULT_THRESHOLD = 0.72


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def save_photo(employee_id: str, image_bytes: bytes) -> str:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    path = FACES_DIR / f"{employee_id}.jpg"
    path.write_bytes(image_bytes)
    return str(path)


def _load_encodings() -> Dict[str, dict]:
    if ENCODINGS_FILE.exists():
        try:
            return json.loads(ENCODINGS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_encodings(data: Dict[str, dict]):
    ENCODINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENCODINGS_FILE.write_text(json.dumps(data))


def _normalize_enc(enc) -> Optional[dict]:
    """Accept both the new {engine, vec} form and a legacy bare list
    (which was always a Facenet embedding)."""
    if isinstance(enc, dict) and "vec" in enc:
        return {"engine": enc.get("engine", "facenet"), "vec": enc["vec"]}
    if isinstance(enc, list):
        return {"engine": "facenet", "vec": enc}
    return None


def save_encoding(employee_id: str, embedding: dict):
    enc = _load_encodings()
    enc[employee_id] = {"engine": embedding["engine"], "vec": embedding["vec"]}
    _save_encodings(enc)


def delete_encoding(employee_id: str):
    enc = _load_encodings()
    enc.pop(employee_id, None)
    _save_encodings(enc)


# ---------------------------------------------------------------------------
# Embedding engines
# ---------------------------------------------------------------------------

def _facenet_embedding(image_bytes: bytes) -> Optional[List[float]]:
    """128-d Facenet embedding via DeepFace, or None if unavailable/failed."""
    tmp_path = None
    try:
        from deepface import DeepFace
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name
        result = DeepFace.represent(
            img_path=tmp_path,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv",
        )
        if result:
            return result[0]["embedding"]
    except Exception as exc:
        print(f"[bio] DeepFace unavailable, using OpenCV fallback: {exc}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return None


_CASCADE = None


def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _CASCADE


def _opencv_embedding(image_bytes: bytes) -> Optional[dict]:
    """Lightweight, TensorFlow-free face embedding: detect the largest face,
    crop, normalise to a fixed-size equalised grayscale vector. Deterministic —
    the same image always yields the same vector (so it verifies against itself).
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_found = False
    try:
        faces = _get_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
    except Exception:
        faces = []

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.12 * w)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        crop = gray[y0:y1, x0:x1]
        face_found = True
    else:
        crop = gray  # whole image — still lets identical photos match

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (120, 120))
    crop = cv2.equalizeHist(crop)
    v = crop.astype(np.float32).flatten()
    v -= v.mean()
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        return None
    return {"engine": "opencv", "vec": (v / norm).tolist(), "face": face_found}


def compute_embedding(image_bytes: bytes) -> Optional[dict]:
    """Return an engine-tagged embedding dict {engine, vec, face} or None.

    Tries Facenet first; falls back to the OpenCV embedding when DeepFace is
    unavailable. Returns None only if the image can't be decoded at all.
    """
    vec = _facenet_embedding(image_bytes)
    if vec is not None:
        return {"engine": "facenet", "vec": vec, "face": True}
    return _opencv_embedding(image_bytes)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_face(image_bytes: bytes, persons: List[Dict]) -> Dict:
    query = compute_embedding(image_bytes)
    if query is None:
        return {"matched": False, "confidence": 0.0, "person": None,
                "engine": "none"}

    encodings   = _load_encodings()
    best_sim    = 0.0
    best_person = None

    for p in persons:
        enc = _normalize_enc(encodings.get(p["employee_id"]))
        if enc is None or enc["engine"] != query["engine"]:
            continue  # only compare like-with-like engines
        sim = _cosine_sim(query["vec"], enc["vec"])
        if sim > best_sim:
            best_sim, best_person = sim, p

    threshold = THRESHOLDS.get(query["engine"], DEFAULT_THRESHOLD)
    matched   = best_sim >= threshold and best_person is not None
    engine_label = "DeepFace/Facenet" if query["engine"] == "facenet" else "OpenCV/face-crop"

    return {
        "matched":    matched,
        "confidence": round(best_sim, 4),
        "person":     best_person if matched else None,
        "engine":     engine_label,
    }
