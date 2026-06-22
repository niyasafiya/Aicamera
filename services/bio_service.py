"""
Face encoding and verification via DeepFace (Facenet).
Encodings stored as JSON; photos stored under data/faces/.
"""
from __future__ import annotations

import json
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

FACES_DIR      = Path("data/faces")
ENCODINGS_FILE = Path("data/encodings.json")


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def save_photo(employee_id: str, image_bytes: bytes) -> str:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    path = FACES_DIR / f"{employee_id}.jpg"
    path.write_bytes(image_bytes)
    return str(path)


def _load_encodings() -> Dict[str, List[float]]:
    if ENCODINGS_FILE.exists():
        try:
            return json.loads(ENCODINGS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_encodings(data: Dict[str, List[float]]):
    ENCODINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENCODINGS_FILE.write_text(json.dumps(data))


def save_encoding(employee_id: str, embedding: List[float]):
    enc = _load_encodings()
    enc[employee_id] = embedding
    _save_encodings(enc)


def delete_encoding(employee_id: str):
    enc = _load_encodings()
    enc.pop(employee_id, None)
    _save_encodings(enc)


# ---------------------------------------------------------------------------
# Face embedding
# ---------------------------------------------------------------------------

def compute_embedding(image_bytes: bytes) -> Optional[List[float]]:
    """Return 128-d Facenet embedding, or None on failure."""
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
        print(f"[DeepFace] Embedding error: {exc}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return None


def _cosine_sim(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

MATCH_THRESHOLD = 0.72


def verify_face(image_bytes: bytes, persons: List[Dict]) -> Dict:
    embedding = compute_embedding(image_bytes)
    if embedding is None:
        return {
            "matched":    False,
            "confidence": 0.0,
            "person":     None,
            "engine":     "DeepFace/Facenet",
        }

    encodings    = _load_encodings()
    best_sim     = 0.0
    best_person  = None

    for p in persons:
        eid = p["employee_id"]
        if eid in encodings:
            sim = _cosine_sim(embedding, encodings[eid])
            if sim > best_sim:
                best_sim    = sim
                best_person = p

    matched = best_sim >= MATCH_THRESHOLD and best_person is not None
    return {
        "matched":    matched,
        "confidence": round(best_sim, 4),
        "person":     best_person if matched else None,
        "engine":     "DeepFace/Facenet",
    }
