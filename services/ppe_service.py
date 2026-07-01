"""
PPE compliance detection — hard-hat / safety-vest.

Ported from the team's app/services/ppe_detector.py and adapted to this
project's flat structure (no SQLAlchemy, no per-camera DB writes here — the
router decides what to log).

Pipeline (same idea as the original):
  1. Detect people with the general yolov8n COCO model (class 0 = person).
     Reliable full-body boxes in any scene.
  2. Run a PPE model at low confidence to gather hat / no-hat (and vest /
     no-vest, if the model exposes those classes) signals.
  3. Per person, decide compliance by overlapping the PPE signal boxes with
     the person's head region (hat) or torso region (vest).

The PPE model auto-downloads from HuggingFace on first use and is cached to
models/ppe_model.pt. Everything degrades gracefully if ultralytics or the model
is unavailable.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    _LIBS_AVAILABLE = True
except ImportError:  # pragma: no cover
    log.warning("ultralytics/opencv not installed — PPE detection disabled")
    _LIBS_AVAILABLE = False

# Cache the downloaded PPE model alongside the project's other models.
_MODELS_DIR = Path("models")
_PPE_MODEL_FILE = _MODELS_DIR / "ppe_model.pt"
# Hard-hat detector trained on the construction-safety dataset.
_HF_REPO = "keremberke/yolov8s-hard-hat-detection"
_HF_FILE = "best.pt"
# General COCO model already shipped with this project (used for person boxes).
_PERSON_MODEL_FILE = "models/yolov8n.pt"

# Label vocabularies — matched case-insensitively against the model's class names.
_HARDHAT_LABELS = {"hardhat", "helmet", "hard hat", "hard-hat", "hard_hat"}
_VEST_LABELS = {"safety vest", "vest", "hi-vis", "hiviz", "safety-vest", "safety_vest"}
_NO_HARDHAT_LABELS = {
    "no-hardhat", "no hardhat", "no-helmet", "no helmet",
    "without helmet", "without hardhat",
}
_NO_VEST_LABELS = {
    "no-vest", "no vest", "no-safety vest", "no safety vest",
    "no-safety-vest", "without vest",
}


# ---------------------------------------------------------------------------
# Geometry helpers (ported verbatim in spirit)
# ---------------------------------------------------------------------------

def _frame_zone(xyxy, frame_shape) -> str:
    """Human-readable zone label from the box-centre position."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    col = "Left" if cx < w / 3 else ("Right" if cx > 2 * w / 3 else "Centre")
    row = "Top" if cy < h / 3 else ("Bottom" if cy > 2 * h / 3 else "")
    return f"{row} {col}".strip() if row else col


def _box_overlap(a, b) -> float:
    """Intersection area / area of box a (how much of a is covered by b)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def _download_ppe_model() -> Optional[str]:
    if _PPE_MODEL_FILE.exists():
        return str(_PPE_MODEL_FILE)
    log.info("Downloading PPE detection model from HuggingFace…")
    try:
        import shutil
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, _PPE_MODEL_FILE)
        log.info("PPE model saved to %s", _PPE_MODEL_FILE)
        return str(_PPE_MODEL_FILE)
    except Exception as exc:
        log.warning("PPE model download failed (%s) — hat/vest flags disabled, "
                    "person boxes still work", exc)
        return None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class PPEDetector:
    def __init__(self):
        self.model = None          # PPE model (hat/vest classes)
        self._person_model = None  # general COCO model (person boxes)
        self._has_hat = False
        self._has_vest = False
        if not _LIBS_AVAILABLE:
            return

        try:
            self._person_model = YOLO(_PERSON_MODEL_FILE)
        except Exception as exc:
            log.warning("Person model unavailable (%s) — PPE detection disabled", exc)
            return

        model_path = _download_ppe_model()
        if model_path:
            try:
                self.model = YOLO(model_path)
                classes = {v.lower() for v in self.model.names.values()}
                self._has_hat = bool(
                    classes & (_HARDHAT_LABELS | _NO_HARDHAT_LABELS)
                )
                self._has_vest = bool(
                    classes & (_VEST_LABELS | _NO_VEST_LABELS)
                )
                log.info("PPE model classes=%s (hat=%s vest=%s)",
                         list(self.model.names.values()), self._has_hat, self._has_vest)
            except Exception as exc:
                log.warning("PPE model load failed (%s)", exc)
                self.model = None

    @property
    def available(self) -> bool:
        return self._person_model is not None

    # ------------------------------------------------------------------
    def analyze_frame(self, frame) -> dict:
        """Return per-person boxes with compliance status and a violation summary."""
        empty = {"boxes": [], "violations": [], "has_violation": False,
                 "max_conf": 0.0, "persons": []}
        if not self.available:
            return empty

        boxes, violations = self._person_crosscheck(frame)
        max_conf = max((b["conf"] for b in boxes), default=0.0)
        persons = [
            {"location": b["location"],
             "status": "violation" if b["violation"] else "compliant",
             "missing": b.get("missing", []),
             "conf": round(b["conf"], 2)}
            for b in boxes
        ]
        return {
            "boxes": boxes,
            "violations": sorted(violations),
            "has_violation": bool(violations),
            "max_conf": max_conf,
            "persons": persons,
        }

    def _person_crosscheck(self, frame) -> Tuple[List[dict], Set[str]]:
        h_frame, w_frame = frame.shape[:2]
        min_box_h = h_frame * 0.03  # ignore only tiny detections (<3% of frame)

        # Larger inference size + lower confidence → catch people the default misses:
        # small/distant, partially occluded, and those cut off at the frame edge
        # (e.g. the person nearest the camera at the bottom). imgsz=960 on yolov8n
        # is still fast enough while noticeably improving recall.
        pr = self._person_model(frame, conf=0.25, iou=0.5, classes=[0],
                                imgsz=960, max_det=100, verbose=False)[0]

        raw: List[Tuple[list, float]] = []
        for b in pr.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            box_h, box_w = y2 - y1, x2 - x1
            if box_h < min_box_h:
                continue
            # Very relaxed aspect gate — allow seated / crouching / edge-cropped
            # people; reject only clearly horizontal blobs (h/w < 0.28).
            if box_w > 0 and (box_h / box_w) < 0.28:
                continue
            raw.append(([x1, y1, x2, y2], float(b.conf)))

        # Greedy NMS — drop boxes that overlap a stronger one by >50%
        raw.sort(key=lambda t: t[1], reverse=True)
        kept: List[Tuple[list, float]] = []
        for box, conf in raw:
            if not any(_box_overlap(box, k[0]) > 0.5 for k in kept):
                kept.append((box, conf))

        persons = [k[0] for k in kept]
        person_confs = [k[1] for k in kept]
        if not persons:
            return [], set()

        # Collect PPE signal boxes (if a PPE model is loaded)
        hat_boxes, no_hat_boxes, vest_boxes, no_vest_boxes = [], [], [], []
        if self.model is not None:
            ppe = self.model(frame, conf=0.05, verbose=False)[0]
            for b in ppe.boxes:
                label = ppe.names[int(b.cls)].lower()
                box = b.xyxy[0].tolist()
                if label in _HARDHAT_LABELS:
                    hat_boxes.append(box)
                elif label in _NO_HARDHAT_LABELS:
                    no_hat_boxes.append(box)
                elif label in _VEST_LABELS:
                    vest_boxes.append(box)
                elif label in _NO_VEST_LABELS:
                    no_vest_boxes.append(box)

        boxes: List[dict] = []
        violations: Set[str] = set()
        for xyxy, conf in zip(persons, person_confs):
            x1, y1, x2, y2 = xyxy
            head = [x1, y1, x2, y1 + (y2 - y1) * 0.45]       # top 45% = head/shoulders
            torso = [x1, y1 + (y2 - y1) * 0.30, x2, y1 + (y2 - y1) * 0.75]
            missing: List[str] = []

            if self._has_hat:
                has_hat = any(_box_overlap(head, hb) > 0.03 for hb in hat_boxes)
                has_no_hat = any(_box_overlap(head, nb) > 0.03 for nb in no_hat_boxes)
                if has_no_hat and not has_hat:
                    missing.append("hard hat")

            if self._has_vest:
                has_vest = any(_box_overlap(torso, vb) > 0.03 for vb in vest_boxes)
                has_no_vest = any(_box_overlap(torso, nb) > 0.03 for nb in no_vest_boxes)
                if has_no_vest and not has_vest:
                    missing.append("safety vest")

            violation = bool(missing)
            violations.update(missing)
            label = ("NO " + " / ".join(missing)) if violation else "PPE OK"
            boxes.append({
                "xyxy": xyxy, "label": label, "conf": conf,
                "violation": violation, "missing": missing,
                "location": _frame_zone(xyxy, frame.shape),
            })

        return boxes, violations


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def draw_boxes(frame, boxes: List[dict]):
    """Red box = violation, green = compliant. Returns an annotated copy."""
    out = frame.copy()
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box["xyxy"]]
        color = (40, 40, 220) if box["violation"] else (40, 200, 80)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box["xyxy"]]
        color = (40, 40, 220) if box["violation"] else (40, 200, 80)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        loc = box.get("location", "")
        label = (f"{box['label']}  {box['conf']:.0%}  [{loc}]"
                 if loc else f"{box['label']}  {box['conf']:.0%}")
        (tw, th), bl = cv2.getTextSize(label, font, 0.55, 1)
        tag_y = max(y1, th + bl + 8)
        cv2.rectangle(out, (x1, tag_y - th - bl - 6), (x1 + tw + 8, tag_y), color, -1)
        cv2.putText(out, label, (x1 + 4, tag_y - bl - 3), font, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_detector: Optional[PPEDetector] = None


def get_ppe_detector() -> PPEDetector:
    global _detector
    if _detector is None:
        _detector = PPEDetector()
    return _detector
