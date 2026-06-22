"""
Biometric face recognition module.
Primary engine : DeepFace (auto-downloads models).
Fallback engine: OpenCV LBPH if DeepFace is unavailable.
"""
import cv2
import numpy as np
import base64
import os
import tempfile
from pathlib import Path

FACE_DIR = Path(__file__).parent.parent / "faces"
FACE_DIR.mkdir(exist_ok=True)

# Similarity threshold: above this = match
MATCH_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _decode_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")
    return img


def _decode_b64(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return _decode_bytes(base64.b64decode(b64_str))


def _face_path(employee_id: str) -> Path:
    safe = employee_id.replace("/", "_").replace("\\", "_")
    return FACE_DIR / f"{safe}.jpg"


# ---------------------------------------------------------------------------
# Face detection (to validate that a photo contains a face)
# ---------------------------------------------------------------------------

_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _cascade


def detect_faces_in_image(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    cascade = _get_cascade()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60)
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# ---------------------------------------------------------------------------
# DeepFace wrapper
# ---------------------------------------------------------------------------

def _deepface_compare(img_bgr: np.ndarray, stored_path: Path) -> float:
    """Return similarity score 0–1 using DeepFace Facenet model."""
    from deepface import DeepFace  # imported lazily

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    try:
        result = DeepFace.verify(
            img1_path=tmp_path,
            img2_path=str(stored_path),
            model_name="Facenet",
            enforce_detection=False,
            silent=True,
        )
        distance = result.get("distance", 1.0)
        # Facenet distance is cosine; 0 = identical, ~0.4 = threshold
        # Convert to 0–1 similarity: clamp and invert
        similarity = max(0.0, 1.0 - (distance / 0.8))
        return round(min(similarity, 1.0), 4)
    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OpenCV LBPH fallback
# ---------------------------------------------------------------------------

def _histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple greyscale histogram correlation as fallback similarity."""
    def hist(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (128, 128))
        h = cv2.calcHist([g], [0], None, [256], [0, 256])
        cv2.normalize(h, h)
        return h

    corr = cv2.compareHist(hist(img1), hist(img2), cv2.HISTCMP_CORREL)
    return max(0.0, round(float(corr), 4))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_face(employee_id: str, img_bgr: np.ndarray) -> dict:
    """
    Save the face image for a registered person.
    Returns {'ok': bool, 'face_detected': bool, 'path': str}.
    """
    faces = detect_faces_in_image(img_bgr)
    face_detected = len(faces) > 0

    # Crop to largest face if detected, else save full image
    if face_detected:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(max(w, h) * 0.20)
        ih, iw = img_bgr.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad)
        y2 = min(ih, y + h + pad)
        face_crop = img_bgr[y1:y2, x1:x2]
    else:
        face_crop = img_bgr

    path = _face_path(employee_id)
    face_crop = cv2.resize(face_crop, (224, 224))
    cv2.imwrite(str(path), face_crop)

    return {"ok": True, "face_detected": face_detected, "path": str(path)}


def verify_face(img_bgr: np.ndarray, persons: list[dict]) -> dict:
    """
    Compare query image against all registered persons.
    persons: list of dicts with keys id, name, employee_id, department, clearance_level.
    Returns best match result dict.
    """
    best_score = 0.0
    best_person = None
    use_deepface = True

    # Test if deepface is available
    try:
        from deepface import DeepFace  # noqa: F401
    except ImportError:
        use_deepface = False

    for person in persons:
        stored = _face_path(person["employee_id"])
        if not stored.exists():
            continue

        if use_deepface:
            try:
                score = _deepface_compare(img_bgr, stored)
            except Exception:
                score = _histogram_similarity(img_bgr, cv2.imread(str(stored)))
        else:
            stored_img = cv2.imread(str(stored))
            if stored_img is None:
                continue
            score = _histogram_similarity(img_bgr, stored_img)

        if score > best_score:
            best_score = score
            best_person = person

    matched = best_score >= MATCH_THRESHOLD and best_person is not None
    decision = "GRANTED" if matched else "DENIED"

    return {
        "matched": matched,
        "confidence": round(best_score, 3),
        "decision": decision,
        "person": best_person if matched else None,
        "engine": "DeepFace/Facenet" if use_deepface else "OpenCV-Histogram",
    }
