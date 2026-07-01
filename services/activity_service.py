"""
Worker activity classification — working / on-phone / chatting / resting.

Ported from the team's app/services/activity_detector.py and adapted to this
project. Two signals:

  * Pose (yolov8n-pose): a slumped/short torso → "resting".
  * Proximity: two people whose centres are close → "chatting".

Added here (not in the original) to answer "working vs. on phone":
  * Cell-phone proximity: a COCO 'cell phone' box near a person's upper body →
    "on phone". Uses the same yolov8n COCO model the PPE service already loads.

Person boxes are passed in by the caller (the PPE service already detects them),
so this service only adds the pose + phone passes.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional

log = logging.getLogger(__name__)

try:
    import numpy as np
    from ultralytics import YOLO
    _LIBS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIBS_AVAILABLE = False

_POSE_MODEL_FILE = "yolov8n-pose.pt"     # ultralytics auto-downloads if missing
_PHONE_MODEL_FILE = "models/yolov8n.pt"  # COCO model already in this project
_COCO_CELL_PHONE = 67                    # COCO class id for "cell phone"

# COCO pose keypoint indices
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12

_CHATTING_DIST_FACTOR = 1.15  # proximity threshold × avg person width (tighter = fewer false pairs)
_RESTING_TORSO_RATIO = 0.30   # torso / box-height below this → resting
_PHONE_CONF = 0.40            # min confidence for a cell-phone detection
_POSE_CONF = 0.30             # min confidence for pose keypoints


def _iou_fraction(a, b) -> float:
    """Fraction of box-a area covered by box-b."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1, (ax2 - ax1) * (ay2 - ay1))


class ActivityDetector:
    """Classify each detected person as working / on-phone / chatting / resting."""

    def __init__(self):
        self._pose = None
        self._phone = None
        if not _LIBS_AVAILABLE:
            return
        try:
            self._pose = YOLO(_POSE_MODEL_FILE)
            log.info("Activity detector: pose model loaded (%s)", _POSE_MODEL_FILE)
        except Exception as exc:
            log.warning("Activity detector: pose model unavailable (%s) — "
                        "geometry-only resting detection", exc)
        try:
            self._phone = YOLO(_PHONE_MODEL_FILE)
        except Exception as exc:
            log.warning("Activity detector: phone model unavailable (%s) — "
                        "on-phone detection disabled", exc)

    # ------------------------------------------------------------------
    def classify(self, frame, persons_xyxy: list, locations: list) -> List[dict]:
        """Return activity-event dicts for the API/annotation layers.

        Each person resolves to exactly one primary state in priority order:
        on-phone > chatting > resting > working. Chatting also emits a pair event.
        """
        n = len(persons_xyxy)
        if n == 0:
            return []

        activities = ["working"] * n

        # ── Resting (pose) ───────────────────────────────────────────
        if self._pose is not None:
            try:
                pose_res = self._pose(frame, conf=_POSE_CONF, verbose=False)[0]
                kpts_data = (pose_res.keypoints.data.cpu().numpy()
                             if pose_res.keypoints is not None else [])
                for pi, pbox in enumerate(persons_xyxy):
                    kpts = self._match_kpts(pbox, pose_res.boxes, kpts_data)
                    if kpts is not None and self._is_resting(kpts, pbox):
                        activities[pi] = "resting"
            except Exception:
                log.debug("Pose inference failed — skipping resting detection")
        else:
            for pi, (x1, y1, x2, y2) in enumerate(persons_xyxy):
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                if w / h > 0.65:
                    activities[pi] = "resting"

        # ── Chatting (proximity) ─────────────────────────────────────
        chat_pairs: List[tuple] = []
        for i in range(n):
            for j in range(i + 1, n):
                if self._are_close(persons_xyxy[i], persons_xyxy[j]):
                    chat_pairs.append((i, j))
        for i, j in chat_pairs:
            for idx in (i, j):
                if activities[idx] == "working":  # don't override resting
                    activities[idx] = "chatting"

        # ── On phone (cell-phone proximity) ──────────────────────────
        phone_idx = self._phone_users(frame, persons_xyxy)
        for idx in phone_idx:
            activities[idx] = "on phone"   # highest priority signal

        # ── Build events ─────────────────────────────────────────────
        events: List[dict] = []
        for pi, activity in enumerate(activities):
            loc = locations[pi] if pi < len(locations) else "Unknown"
            if activity != "chatting":   # chatting reported as pairs below
                events.append({"type": activity, "person": pi + 1, "location": loc})

        seen: set = set()
        for pi, pj in chat_pairs:
            # only a genuine chat if neither got upgraded to on-phone
            if activities[pi] != "chatting" and activities[pj] != "chatting":
                continue
            key = (min(pi, pj), max(pi, pj))
            if key in seen:
                continue
            seen.add(key)
            loc_i = locations[pi] if pi < len(locations) else "Unknown"
            loc_j = locations[pj] if pj < len(locations) else "Unknown"
            loc = loc_i if loc_i == loc_j else f"{loc_i} / {loc_j}"
            events.append({"type": "chatting", "person_a": pi + 1,
                           "person_b": pj + 1, "location": loc})
        return events

    # ------------------------------------------------------------------
    def _phone_users(self, frame, persons_xyxy: list) -> set:
        if self._phone is None or not persons_xyxy:
            return set()
        try:
            res = self._phone(frame, conf=_PHONE_CONF, classes=[_COCO_CELL_PHONE],
                              verbose=False)[0]
        except Exception:
            return set()
        phones = [b.xyxy[0].tolist() for b in res.boxes] if res.boxes is not None else []
        users: set = set()
        for ph in phones:
            pcx = (ph[0] + ph[2]) / 2
            pcy = (ph[1] + ph[3]) / 2
            ph_area = max(1.0, (ph[2] - ph[0]) * (ph[3] - ph[1]))
            # Assign the phone to the person whose hand/upper-body region best contains it
            best, best_i = 0.0, -1
            for pi, (x1, y1, x2, y2) in enumerate(persons_xyxy):
                p_area = max(1.0, (x2 - x1) * (y2 - y1))
                # A "phone" bigger than a quarter of the person is almost certainly a
                # false positive (screen, box, etc.) — skip it.
                if ph_area > p_area * 0.25:
                    continue
                # Held phones sit in the upper ¾ of the body (head → hands), not at the feet
                region = [x1, y1, x2, y1 + (y2 - y1) * 0.75]
                if region[0] <= pcx <= region[2] and region[1] <= pcy <= region[3]:
                    cov = _iou_fraction(ph, region)
                    if cov > best:
                        best, best_i = cov, pi
            if best_i >= 0 and best > 0.02:
                users.add(best_i)
        return users

    def _match_kpts(self, pbox, pose_boxes, kpts_data):
        if len(kpts_data) == 0:
            return None
        best, best_i = 0.30, -1
        for pi, pb in enumerate(pose_boxes):
            overlap = _iou_fraction(pbox, pb.xyxy[0].tolist())
            if overlap > best:
                best, best_i = overlap, pi
        return kpts_data[best_i] if 0 <= best_i < len(kpts_data) else None

    def _is_resting(self, kpts, pbox) -> bool:
        _, y1, _, y2 = pbox
        box_h = max(1, y2 - y1)

        def y_of(idx):
            return float(kpts[idx][1]) if kpts[idx][2] > 0.25 else None

        shoulder_y = next(filter(None, (y_of(_L_SHOULDER), y_of(_R_SHOULDER))), None)
        hip_y = next(filter(None, (y_of(_L_HIP), y_of(_R_HIP))), None)
        if shoulder_y is not None and hip_y is not None:
            return (hip_y - shoulder_y) / box_h < _RESTING_TORSO_RATIO
        return False

    def _are_close(self, a, b) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        cx_a, cy_a = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        cx_b, cy_b = (bx1 + bx2) / 2, (by1 + by2) / 2
        wa, wb = max(1, ax2 - ax1), max(1, bx2 - bx1)
        ha, hb = max(1, ay2 - ay1), max(1, by2 - by1)
        avg_w, avg_h = (wa + wb) / 2, (ha + hb) / 2
        # Both people must be at a similar scale (similar depth) — a near + far
        # pair that merely overlaps in 2D isn't actually chatting.
        if not (0.5 <= wa / wb <= 2.0):
            return False
        # …and at a similar ground level (standing side by side), not stacked.
        if abs(cy_a - cy_b) > avg_h * 0.5:
            return False
        dist = math.hypot(cx_a - cx_b, cy_a - cy_b)
        return dist < avg_w * _CHATTING_DIST_FACTOR


_detector: Optional[ActivityDetector] = None


def get_activity_detector() -> ActivityDetector:
    global _detector
    if _detector is None:
        _detector = ActivityDetector()
    return _detector
