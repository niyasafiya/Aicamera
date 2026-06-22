"""
Live demo pipeline — YardMonitor-style threaded vehicle detection.

Reads a demo video in a loop, runs YOLOv8+ByteTrack, performs non-blocking
EasyOCR on the best plate crop per tracked vehicle, checks authorization,
and serves annotated JPEG frames for all 4 camera MJPEG streams.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import db

# ---------------------------------------------------------------------------
# Constants  (mirror YardMonitor defaults)
# ---------------------------------------------------------------------------
_DIRECTION_LINE_Y = 0.55   # fraction of frame height for virtual gate line
_BANNER_DURATION  = 3.5    # seconds to show access banner
_OCR_MIN_SHARP    = 12.0   # minimum Laplacian variance before running OCR
_OCR_MIN_INTERVAL = 0.8    # seconds between OCR retries per track
_OCR_CONF_STOP    = 0.80   # stop retrying once confidence reaches this level

_AUTHORIZED_COLOR   = (40,  201, 168)  # teal — granted
_UNAUTHORIZED_COLOR = (242,  86, 110)  # red  — denied
_UNKNOWN_COLOR      = ( 77, 124, 254)  # blue — pending OCR
_LINE_COLOR         = (240, 169,  59)  # amber — gate line

_CAM_LABELS: Dict[str, Tuple[str, str]] = {
    "cam01": ("CAM-01", "Main Gate — In"),
    "cam12": ("CAM-12", "Loading Bay 2"),
    "cam08": ("CAM-08", "Warehouse Aisle 4"),
    "cam05": ("CAM-05", "Turnstile B"),
}


# ---------------------------------------------------------------------------
# Per-track state  (mirrors YardMonitor's _TrackState)
# ---------------------------------------------------------------------------
@dataclass
class _TrackState:
    best_crop:       Optional[np.ndarray] = None
    best_sharp:      float                = -1.0
    plate:           str                  = ""
    confidence:      float                = 0.0
    authorized:      Optional[bool]       = None
    owner:           str                  = ""
    ocr_future                            = None   # concurrent.futures.Future
    ocr_submit_time: float                = 0.0
    last_y:          int                  = 0
    crossed:         bool                 = False
    gate_triggered:  bool                 = False


# ---------------------------------------------------------------------------
# OCR worker — runs in thread pool, fully self-contained, no shared state
# ---------------------------------------------------------------------------
def _ocr_task(crop: np.ndarray) -> Tuple[str, float, Optional[bool], str]:
    """Returns (plate, confidence, authorized, owner). Safe to run in any thread."""
    from services.anpr_service import read_plate_crop
    result = read_plate_crop(crop)
    if not result or not result.get("plate"):
        return "", 0.0, None, ""

    plate = result["plate"]
    conf  = result["confidence"]
    norm  = plate.upper().replace(" ", "").replace("-", "")

    try:
        conn = db.get_conn()
        row  = conn.execute(
            "SELECT owner FROM authorized_vehicles WHERE plate=?", (norm,)
        ).fetchone()
        conn.close()
        authorized = row is not None
        owner      = row["owner"] if row else ""
    except Exception:
        authorized, owner = False, ""

    return plate, conf, authorized, owner


# ---------------------------------------------------------------------------
# Demo pipeline
# ---------------------------------------------------------------------------
class DemoPipeline(threading.Thread):
    """
    Background thread that processes a demo video and serves annotated frames.
    Modelled on YardMonitor's CameraPipeline:
      - YOLOv8 + ByteTrack for stable track IDs
      - Laplacian-sharpness best-crop selection
      - Non-blocking EasyOCR via ThreadPoolExecutor
      - Direction-line crossing → ACCESS GRANTED / DENIED banner
    """

    def __init__(self, video_path: str):
        super().__init__(daemon=True, name="DemoPipeline")
        self._path     = video_path
        self._filename = Path(video_path).name
        self._running  = False
        self._backend  = "unknown"

        self._frames: Dict[str, bytes] = {}
        self._events: deque            = deque(maxlen=50)
        self._banner: dict             = {}
        self._lock                     = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def backend(self) -> str:
        return self._backend

    def stop(self):
        self._running = False

    def get_frame(self, cam_id: str) -> Optional[bytes]:
        with self._lock:
            return self._frames.get(cam_id)

    def get_recent_events(self, n: int = 20) -> List[dict]:
        with self._lock:
            evs = list(self._events)
        return evs[::-1][:n]

    # ── Thread body ──────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        from services.yolo_service import get_detector, find_plate_contour

        detector      = get_detector()
        self._backend = detector.backend
        executor      = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PipeOCR")
        track_states: Dict[int, _TrackState] = {}

        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            print(f"[Pipeline] Cannot open: {self._path}")
            self._running = False
            return

        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delay = 1.0 / min(fps, 30.0)
        print(f"[Pipeline] Started  file={self._filename}  backend={self._backend}  fps={fps:.1f}")

        while self._running:
            t0     = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                track_states.clear()
                continue

            h, w   = frame.shape[:2]
            line_y = int(h * _DIRECTION_LINE_Y)
            now    = time.time()

            use_track = (detector.backend == "yolov8")
            dets = detector.detect(frame, track=use_track) if detector.available else []

            for d in dets:
                tid = d.track_id if d.track_id is not None else id(d)

                if tid not in track_states:
                    track_states[tid] = _TrackState(last_y=d.center[1])

                st = track_states[tid]
                _cx, cy = d.center
                x1, y1, x2, y2 = d.bbox

                # ── Best crop (Laplacian sharpness — YardMonitor) ───────────
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size > 0:
                    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if sharp > st.best_sharp:
                        st.best_sharp = sharp
                        try:
                            px1, py1, px2, py2 = find_plate_contour(crop)
                            psub = crop[py1:py2, px1:px2]
                            st.best_crop = psub.copy() if psub.size > 0 else crop.copy()
                        except Exception:
                            st.best_crop = crop.copy()

                # ── Non-blocking OCR (rate-limited — YardMonitor pattern) ────
                if (
                    st.best_crop is not None
                    and st.best_sharp >= _OCR_MIN_SHARP
                    and st.confidence < _OCR_CONF_STOP
                    and (st.ocr_future is None or st.ocr_future.done())
                    and now - st.ocr_submit_time > _OCR_MIN_INTERVAL
                ):
                    snap              = st.best_crop.copy()
                    st.ocr_submit_time = now
                    st.ocr_future      = executor.submit(_ocr_task, snap)

                # ── Collect OCR result ────────────────────────────────────
                if st.ocr_future is not None and st.ocr_future.done():
                    try:
                        plate, conf, authorized, owner = st.ocr_future.result()
                        if plate and conf > st.confidence:
                            st.plate      = plate
                            st.confidence = conf
                            st.authorized = authorized
                            st.owner      = owner
                    except Exception as exc:
                        print(f"[Pipeline] OCR result error: {exc}")
                    st.ocr_future = None

                # ── Direction-line crossing ──────────────────────────────
                last_y = st.last_y
                if not st.crossed:
                    direction = None
                    if last_y < line_y <= cy:
                        direction  = "IN"
                        st.crossed = True
                    elif last_y > line_y >= cy:
                        direction  = "OUT"
                        st.crossed = True

                    if direction and st.plate and not st.gate_triggered:
                        st.gate_triggered = True
                        self._emit_event(
                            st.plate, st.confidence,
                            st.authorized, st.owner,
                            direction, d.cls_name,
                        )
                        self._set_banner(st.plate, st.authorized)

                st.last_y = cy

            # ── Draw + store per camera ──────────────────────────────────────
            for cam_id, (cam_num, cam_name) in _CAM_LABELS.items():
                annotated = self._draw_frame(
                    frame.copy(), dets, track_states, line_y, cam_num, cam_name
                )
                _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self._lock:
                    self._frames[cam_id] = jpg.tobytes()

            elapsed = time.time() - t0
            wait    = frame_delay - elapsed
            if wait > 0:
                time.sleep(wait)

        cap.release()
        executor.shutdown(wait=False)
        self._running = False
        print(f"[Pipeline] Stopped  file={self._filename}")

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw_frame(
        self,
        frame:        np.ndarray,
        dets:         list,
        track_states: Dict[int, _TrackState],
        line_y:       int,
        cam_num:      str,
        cam_name:     str,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Gate line
        cv2.line(frame, (0, line_y), (w, line_y), _LINE_COLOR, 1)
        cv2.putText(frame, "GATE LINE",
                    (w - 90, line_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, _LINE_COLOR, 1)

        for d in dets:
            tid = d.track_id if d.track_id is not None else id(d)
            st  = track_states.get(tid)

            x1, y1, x2, y2 = d.bbox
            if st and st.authorized is True:
                color = _AUTHORIZED_COLOR
            elif st and st.authorized is False:
                color = _UNAUTHORIZED_COLOR
            else:
                color = _UNKNOWN_COLOR

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if st and st.plate:
                badge     = "OK" if st.authorized else "NO"
                top_label = f"{st.plate}  [{badge}]  {st.confidence:.0%}"
            else:
                top_label = f"{d.cls_name.upper()}  #{tid or '?'}"

            (tw, th), _ = cv2.getTextSize(top_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            ly1 = max(y1 - th - 8, 0)
            cv2.rectangle(frame, (x1, ly1), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, top_label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (10, 10, 10), 1)

            if st and st.owner:
                cv2.putText(frame, st.owner[:24], (x1, y2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)

        self._draw_hud(frame, cam_num, cam_name, len(dets))
        self._draw_banner(frame)
        return frame

    def _draw_hud(self, frame: np.ndarray, cam_num: str, cam_name: str, n_dets: int):
        h, w = frame.shape[:2]
        cv2.circle(frame, (12, 14), 5, (0, 200, 155), -1)
        cv2.putText(frame, cam_name, (24, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (195, 215, 235), 1)
        cv2.putText(frame, cam_num, (24, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 130, 170), 1)
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 72, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 200, 220), 1)
        cv2.circle(frame, (w - 80, 30), 4, (220, 40, 40), -1)
        cv2.putText(frame, "REC", (w - 73, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 40, 40), 1)
        cv2.rectangle(frame, (0, h - 26), (w, h), (8, 13, 20), -1)
        cv2.putText(frame,
                    f"Sentinel AI  {self._backend}  vehicles:{n_dets}",
                    (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (55, 95, 130), 1)

    def _draw_banner(self, frame: np.ndarray):
        if not self._banner or time.time() > self._banner.get("expires", 0):
            return
        h, w  = frame.shape[:2]
        text  = self._banner["text"]
        color = self._banner["color"]
        bh    = 48
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bh), (w, h), (10, 14, 20), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (0, h - bh), (6, h), color, -1)
        cv2.putText(frame, text, (14, h - bh // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

    # ── Events ───────────────────────────────────────────────────────────────

    def _emit_event(
        self,
        plate:      str,
        conf:       float,
        authorized: Optional[bool],
        owner:      str,
        direction:  str,
        cls_name:   str,
    ):
        decision = "GRANTED" if authorized else "DENIED"
        event = {
            "ts":         time.strftime("%H:%M:%S"),
            "plate":      plate,
            "confidence": round(conf, 3),
            "authorized": bool(authorized),
            "owner":      owner or "Unknown",
            "direction":  direction,
            "cls":        cls_name.capitalize(),
            "decision":   decision,
        }
        with self._lock:
            self._events.append(event)

        try:
            conn = db.get_conn()
            conn.execute(
                "INSERT INTO anpr_log (plate, confidence, authorized, decision) "
                "VALUES (?,?,?,?)",
                (plate, conf, 1 if authorized else 0, decision),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            print(f"[Pipeline] DB log error: {exc}")

    def _set_banner(self, plate: str, authorized: Optional[bool]):
        if authorized is None:
            return
        if authorized:
            text  = f"ACCESS GRANTED   {plate}"
            color = _AUTHORIZED_COLOR
        else:
            text  = f"ACCESS DENIED    {plate}"
            color = _UNAUTHORIZED_COLOR
        self._banner = {
            "text":    text,
            "color":   color,
            "expires": time.time() + _BANNER_DURATION,
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_pipeline:      Optional[DemoPipeline] = None
_pipeline_lock: threading.Lock         = threading.Lock()


def get_pipeline() -> Optional[DemoPipeline]:
    return _pipeline


def start_pipeline(video_path: str) -> DemoPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline and _pipeline.is_running:
            _pipeline.stop()
            _pipeline.join(timeout=3.0)
        _pipeline = DemoPipeline(video_path)
        _pipeline.start()
    return _pipeline


def stop_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline:
            _pipeline.stop()
        _pipeline = None
