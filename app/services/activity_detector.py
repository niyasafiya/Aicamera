import logging
import math

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from ultralytics import YOLO
    _LIBS_AVAILABLE = True
except ImportError:
    _LIBS_AVAILABLE = False

# COCO pose keypoint indices
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP,      _R_HIP      = 11, 12

_CHATTING_DIST_FACTOR = 1.4   # proximity threshold × avg person width
_RESTING_TORSO_RATIO  = 0.30  # torso / box-height below this → resting


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
    """Classify each detected person as working / chatting / resting."""

    def __init__(self):
        self._pose = None
        if not _LIBS_AVAILABLE:
            return
        try:
            self._pose = YOLO(settings.POSE_MODEL_PATH)
            logger.info("Activity detector: pose model loaded (%s)", settings.POSE_MODEL_PATH)
        except Exception as exc:
            logger.warning("Activity detector: pose model unavailable (%s) — geometry-only mode", exc)

    # ------------------------------------------------------------------
    def classify(self, frame, persons_xyxy: list, locations: list) -> list[dict]:
        """
        Returns a list of activity-event dicts ready for the API response:
          { type: 'working'|'resting'|'chatting',
            person / person_a / person_b: 1-indexed,
            location: str }
        """
        n = len(persons_xyxy)
        if n == 0:
            return []

        activities = ["working"] * n

        # ── Resting detection ─────────────────────────────────────────
        if self._pose is not None:
            try:
                pose_res = self._pose(frame, conf=0.25, verbose=False)[0]
                kpts_data = (
                    pose_res.keypoints.data.cpu().numpy()
                    if pose_res.keypoints is not None else []
                )
                for pi, pbox in enumerate(persons_xyxy):
                    kpts = self._match_kpts(pbox, pose_res.boxes, kpts_data)
                    if kpts is not None and self._is_resting(kpts, pbox):
                        activities[pi] = "resting"
            except Exception:
                logger.debug("Pose inference failed — skipping resting detection")
        else:
            # Fallback: wide bounding box → likely seated/lying
            for pi, (x1, y1, x2, y2) in enumerate(persons_xyxy):
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                if w / h > 0.65:
                    activities[pi] = "resting"

        # ── Chatting detection (proximity) ────────────────────────────
        chat_pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if self._are_close(persons_xyxy[i], persons_xyxy[j]):
                    chat_pairs.append((i, j))

        chatting_idx = {idx for pair in chat_pairs for idx in pair}
        for idx in chatting_idx:
            if activities[idx] == "working":   # don't override resting
                activities[idx] = "chatting"

        # ── Build event list ──────────────────────────────────────────
        events: list[dict] = []
        seen_chat_pairs: set[tuple[int, int]] = set()

        for pi, activity in enumerate(activities):
            loc = locations[pi] if pi < len(locations) else "Unknown"
            if activity == "working":
                events.append({"type": "working",  "person": pi + 1, "location": loc})
            elif activity == "resting":
                events.append({"type": "resting",  "person": pi + 1, "location": loc})

        for pi, pj in chat_pairs:
            key = (min(pi, pj), max(pi, pj))
            if key in seen_chat_pairs:
                continue
            seen_chat_pairs.add(key)
            loc_i = locations[pi] if pi < len(locations) else "Unknown"
            loc_j = locations[pj] if pj < len(locations) else "Unknown"
            loc = loc_i if loc_i == loc_j else f"{loc_i} / {loc_j}"
            events.append({
                "type":     "chatting",
                "person_a": pi + 1,
                "person_b": pj + 1,
                "location": loc,
            })

        return events

    # ------------------------------------------------------------------
    def _match_kpts(self, pbox, pose_boxes, kpts_data):
        if len(kpts_data) == 0:
            return None
        best, best_i = 0.30, -1  # require at least 30% overlap to match
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
        hip_y      = next(filter(None, (y_of(_L_HIP),      y_of(_R_HIP))),      None)
        if shoulder_y is not None and hip_y is not None:
            return (hip_y - shoulder_y) / box_h < _RESTING_TORSO_RATIO
        return False

    def _are_close(self, a, b) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        cx_a, cy_a = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        cx_b, cy_b = (bx1 + bx2) / 2, (by1 + by2) / 2
        dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
        avg_w = ((ax2 - ax1) + (bx2 - bx1)) / 2
        return dist < avg_w * _CHATTING_DIST_FACTOR
