import json
import logging
import os
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.events import PPEViolation

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    _LIBS_AVAILABLE = True
except ImportError:
    logger.warning("ultralytics/opencv not installed — PPE detection disabled")
    _LIBS_AVAILABLE = False

_PPE_MODEL_FILE = "ppe_model.pt"

_VIOLATION_LABELS = {
    "no-hardhat", "no hardhat", "no-helmet", "no helmet",
    "no-vest", "no vest", "no-safety vest", "no safety vest",
    "without helmet", "without hardhat", "without vest",
}
_HARDHAT_LABELS = {"hardhat", "helmet", "hard hat", "hard-hat", "hard_hat"}
_VEST_LABELS    = {"safety vest", "vest", "hi-vis", "hiviz", "safety-vest", "safety_vest"}

_VIOLATION_NAMES = {
    "no-hardhat": "hard hat", "no hardhat": "hard hat",
    "no-helmet":  "hard hat", "no helmet":  "hard hat",
    "without helmet": "hard hat", "without hardhat": "hard hat",
    "no-vest": "safety vest",  "no vest": "safety vest",
    "no-safety vest": "safety vest", "no safety vest": "safety vest",
    "without vest": "safety vest",
}

_cooldowns: dict[str, datetime] = {}


def _download_ppe_model() -> str:
    if os.path.exists(_PPE_MODEL_FILE):
        return _PPE_MODEL_FILE
    logger.info("Downloading PPE detection model from HuggingFace…")
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        cached = hf_hub_download(
            repo_id="keremberke/yolov8s-hard-hat-detection",
            filename="best.pt",
        )
        shutil.copy2(cached, _PPE_MODEL_FILE)
        logger.info(f"PPE model saved to {_PPE_MODEL_FILE}")
        return _PPE_MODEL_FILE
    except Exception as exc:
        logger.warning(f"PPE model download failed ({exc}), falling back to {settings.PPE_MODEL_PATH}")
        return settings.PPE_MODEL_PATH


def _frame_zone(xyxy, frame_shape) -> str:
    """Return a human-readable zone label based on the box centre position."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    col = "Left" if cx < w / 3 else ("Right" if cx > 2 * w / 3 else "Centre")
    row = "Top"   if cy < h / 3 else ("Bottom" if cy > 2 * h / 3 else "")
    return f"{row} {col}".strip() if row else col


def _box_overlap(a, b) -> float:
    """Intersection area / area of box a (how much of a is covered by b)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


class PPEDetector:
    def __init__(self):
        if not _LIBS_AVAILABLE:
            self.model = None
            self._person_model = None
            self._direct_mode = False
            return

        model_path = _download_ppe_model()
        self.model = YOLO(model_path)

        classes = {v.lower() for v in self.model.names.values()}
        self._direct_mode = bool(classes & _VIOLATION_LABELS)

        # General person detector — reliable on any image/context
        self._person_model = YOLO("yolov8n.pt")

        logger.info(f"PPE model: {model_path}  classes={list(self.model.names.values())}")
        logger.info(f"Direct violation mode: {self._direct_mode}")
        os.makedirs(settings.ALERT_IMAGE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    def analyze_frame(self, frame) -> dict:
        """Detect people with full-body boxes, flag those missing hard hats."""
        if not _LIBS_AVAILABLE or self.model is None:
            return {"boxes": [], "violations": [], "has_violation": False, "max_conf": 0.0}

        # Always use person detector for full-body bounding boxes
        boxes, violations = self._person_crosscheck(frame)

        logger.info(f"Detection: {len(boxes)} person(s), violations={list(violations)}")

        max_conf = max((b["conf"] for b in boxes), default=0.0)
        persons = [
            {
                "location":  b["location"],
                "status":    "violation" if b["violation"] else "compliant",
                "conf":      round(b["conf"], 2),
            }
            for b in boxes
        ]
        return {
            "boxes":         boxes,
            "violations":    list(violations),
            "has_violation": bool(violations),
            "max_conf":      max_conf,
            "persons":       persons,
        }

    def _person_crosscheck(self, frame) -> tuple[list, set]:
        """
        Detect people with yolov8n (class 0 only), then use the PPE model's
        explicit 'no-hat' / 'hat' classes to decide per-person compliance.
        Only flags a violation when the model positively detects NO hat.
        """
        h_frame, w_frame = frame.shape[:2]
        min_box_h = h_frame * 0.08  # ignore detections shorter than 8% of frame height

        # class=0 is COCO "person" — no other class is ever returned
        pr = self._person_model(frame, conf=0.6, classes=[0], verbose=False)[0]

        # Filter by minimum height and collect
        raw: list[tuple[list, float]] = []
        for b in pr.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            box_h = y2 - y1
            box_w = x2 - x1
            if box_h < min_box_h:
                continue
            # A person box should be taller than it is wide (or close to square for seated people)
            if box_w > 0 and (box_h / box_w) < 0.5:
                continue
            raw.append(([x1, y1, x2, y2], float(b.conf)))

        # NMS: merge boxes that share more than 35% overlap — catches same person detected twice
        raw.sort(key=lambda t: t[1], reverse=True)
        kept: list[tuple[list, float]] = []
        for box, conf in raw:
            if not any(_box_overlap(box, k[0]) > 0.35 for k in kept):
                kept.append((box, conf))

        persons      = [k[0] for k in kept]
        person_confs = [k[1] for k in kept]

        logger.info(f"Person detector found {len(persons)} person(s)")

        if not persons:
            return [], set()

        # Run PPE model at very low confidence to capture all signals
        ppe_results = self.model(frame, conf=0.05, verbose=False)[0]

        hat_boxes    = []  # confirmed hat present
        no_hat_boxes = []  # confirmed hat absent

        for b in ppe_results.boxes:
            label = ppe_results.names[int(b.cls)].lower()
            box   = b.xyxy[0].tolist()
            if label in _HARDHAT_LABELS:
                hat_boxes.append(box)
            elif label in _VIOLATION_LABELS and any(w in label for w in ("hat", "helmet", "hardhat")):
                no_hat_boxes.append(box)

        logger.info(f"Hat detections: {len(hat_boxes)}  No-hat detections: {len(no_hat_boxes)}")

        boxes: list[dict] = []
        violations: set[str] = set()

        for xyxy, conf in zip(persons, person_confs):
            x1, y1, x2, y2 = xyxy
            # Head region = top 45% of the person bounding box (generous to catch tilted heads)
            head = [x1, y1, x2, y1 + (y2 - y1) * 0.45]

            has_hat    = any(_box_overlap(head, h) > 0.03 for h in hat_boxes)
            has_no_hat = any(_box_overlap(head, h) > 0.03 for h in no_hat_boxes)
            location   = _frame_zone(xyxy, frame.shape)

            if has_no_hat and not has_hat:
                boxes.append({"xyxy": xyxy, "label": "NO Hardhat", "conf": conf, "violation": True,  "location": location})
                violations.add("hard hat")
            else:
                boxes.append({"xyxy": xyxy, "label": "Hardhat OK", "conf": conf, "violation": False, "location": location})

        return boxes, violations

    # ------------------------------------------------------------------
    def process(self, frame, camera_id: str, db: Session) -> None:
        if not _LIBS_AVAILABLE or self.model is None:
            return

        analysis = self.analyze_frame(frame)
        if not analysis["has_violation"]:
            return

        now = datetime.utcnow()
        last = _cooldowns.get(camera_id)
        if last and (now - last).total_seconds() < settings.ALERT_COOLDOWN_SECONDS:
            return
        _cooldowns[camera_id] = now

        frame_path = _save_frame(frame, camera_id)
        db.add(PPEViolation(
            camera_id=camera_id,
            timestamp=now,
            missing_ppe=json.dumps(analysis["violations"]),
            confidence=analysis["max_conf"] or 0.5,
            frame_path=frame_path,
        ))
        db.commit()
        logger.info(f"PPE violation logged — camera {camera_id} — missing: {analysis['violations']}")


def _save_frame(frame, camera_id: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(settings.ALERT_IMAGE_DIR, f"ppe_{camera_id}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path
