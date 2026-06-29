import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.events import Camera, FallEvent, PPEViolation, ZoneIntrusion
from app.services.fall_detector import FallDetector
from app.services.ppe_detector import PPEDetector
from app.services.zone_monitor import ZoneMonitor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/safety", tags=["video-analysis"])

_executor = ThreadPoolExecutor(max_workers=2)

# Detector singletons — created once on first use
_ppe: PPEDetector | None = None
_zone: ZoneMonitor | None = None
_fall: FallDetector | None = None


def _get_detectors():
    global _ppe, _zone, _fall
    if _ppe is None:
        _ppe = PPEDetector()
        _zone = ZoneMonitor()
        _fall = FallDetector()
    return _ppe, _zone, _fall


def _ppe_process_from_analysis(ppe_det: PPEDetector, analysis: dict, frame, camera_id: str, db: Session) -> None:
    """Apply cooldown + DB save using a pre-computed analysis result."""
    from datetime import datetime
    import json
    from app.services.ppe_detector import _cooldowns, _save_frame
    from app.core.config import settings as _s

    if not analysis["has_violation"]:
        return
    now = datetime.utcnow()
    last = _cooldowns.get(camera_id)
    if last and (now - last).total_seconds() < _s.ALERT_COOLDOWN_SECONDS:
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


def _run_video(content: bytes, suffix: str, camera_id: str, db: Session) -> dict:
    try:
        import cv2
    except ImportError:
        raise RuntimeError("OpenCV not available")

    ppe_det, zone_det, fall_det = _get_detectors()

    # Count existing events before analysis so we can report net-new
    before_ppe  = db.query(PPEViolation).filter(PPEViolation.camera_id == camera_id).count()
    before_zone = db.query(ZoneIntrusion).filter(ZoneIntrusion.camera_id == camera_id).count()
    before_fall = db.query(FallEvent).filter(FallEvent.camera_id == camera_id).count()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file — unsupported format?")

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Analyse at most 90 s; sample ~2 frames/sec
        max_frames = min(total_frames, int(fps * 90))
        step       = max(1, int(fps / 2))

        processed = 0
        idx = 0
        best_frame    = None   # frame with most people (for sample annotation)
        best_analysis = None
        best_count    = -1

        while idx < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            # Single inference call — reuse result for both DB and annotation tracking
            analysis = ppe_det.analyze_frame(frame)
            n_people = len(analysis["boxes"])
            if n_people > best_count:
                best_count    = n_people
                best_frame    = frame.copy()
                best_analysis = analysis
            # Pass result into process logic (skip re-running inference inside)
            _ppe_process_from_analysis(ppe_det, analysis, frame, camera_id, db)
            zone_det.process(frame, camera_id, db)
            fall_det.process(frame, camera_id, db)
            processed += 1
            idx += step

        cap.release()

        after_ppe  = db.query(PPEViolation).filter(PPEViolation.camera_id == camera_id).count()
        after_zone = db.query(ZoneIntrusion).filter(ZoneIntrusion.camera_id == camera_id).count()
        after_fall = db.query(FallEvent).filter(FallEvent.camera_id == camera_id).count()

        result = {
            "status": "complete",
            "camera_id": camera_id,
            "frames_analyzed": processed,
            "duration_seconds": round(processed * step / fps, 1),
            "new_detections": {
                "ppe_violations":  after_ppe  - before_ppe,
                "zone_intrusions": after_zone - before_zone,
                "fall_events":     after_fall - before_fall,
            },
        }

        # Attach annotated sample frame so the frontend can show person boxes
        if best_frame is not None and best_analysis is not None:
            import base64 as _b64
            annotated = _draw_boxes(best_frame, best_analysis["boxes"])
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
            result["annotated_image"] = "data:image/jpeg;base64," + _b64.b64encode(buf.tobytes()).decode()
            result["has_violation"] = best_analysis["has_violation"]

        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _draw_boxes(frame, boxes: list[dict]):
    """Draw annotated bounding boxes on a copy of frame. Returns annotated frame."""
    import cv2
    out     = frame.copy()
    overlay = frame.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1

    # Pass 1 — semi-transparent fill inside each box
    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box["xyxy"]]
        color = (40, 40, 220) if box["violation"] else (40, 200, 80)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

    # Pass 2 — solid thick border + label on top of the blended image
    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box["xyxy"]]
        color = (40, 40, 220) if box["violation"] else (40, 200, 80)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        label = f"{box['label']}  {box['conf']:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        tag_y = max(y1, th + baseline + 8)
        cv2.rectangle(out, (x1, tag_y - th - baseline - 6), (x1 + tw + 8, tag_y), color, -1)
        cv2.putText(out, label, (x1 + 4, tag_y - baseline - 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def _run_image(content: bytes, camera_id: str, db: Session) -> dict:
    try:
        import base64
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("OpenCV not available")

    from datetime import datetime
    import json as _json

    ppe_det, zone_det, fall_det = _get_detectors()

    before_ppe  = db.query(PPEViolation).filter(PPEViolation.camera_id == camera_id).count()
    before_zone = db.query(ZoneIntrusion).filter(ZoneIntrusion.camera_id == camera_id).count()
    before_fall = db.query(FallEvent).filter(FallEvent.camera_id == camera_id).count()

    arr   = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — unsupported format?")

    # Single model pass — used for both drawing and DB save
    analysis = ppe_det.analyze_frame(frame)

    # Draw red/green bounding boxes
    annotated = _draw_boxes(frame, analysis["boxes"])

    # Encode annotated image as base64 JPEG
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    # Persist PPE violation directly (avoids running model a second time)
    if analysis["has_violation"]:
        from app.services.ppe_detector import _cooldowns, _save_frame
        now = datetime.utcnow()
        last = _cooldowns.get(camera_id)
        from app.core.config import settings as _s
        if not last or (now - last).total_seconds() >= _s.ALERT_COOLDOWN_SECONDS:
            _cooldowns[camera_id] = now
            frame_path = _save_frame(frame, camera_id)
            db.add(PPEViolation(
                camera_id=camera_id,
                timestamp=now,
                missing_ppe=_json.dumps(analysis["violations"]),
                confidence=analysis["max_conf"] or 0.5,
                frame_path=frame_path,
            ))
            db.commit()

    zone_det.process(frame, camera_id, db)
    fall_det.process(frame, camera_id, db)

    after_ppe  = db.query(PPEViolation).filter(PPEViolation.camera_id == camera_id).count()
    after_zone = db.query(ZoneIntrusion).filter(ZoneIntrusion.camera_id == camera_id).count()
    after_fall = db.query(FallEvent).filter(FallEvent.camera_id == camera_id).count()

    return {
        "status": "complete",
        "camera_id": camera_id,
        "frames_analyzed": 1,
        "duration_seconds": 0,
        "annotated_image": img_b64,
        "has_violation": analysis["has_violation"],
        "violations": analysis["violations"],
        "new_detections": {
            "ppe_violations":  after_ppe  - before_ppe,
            "zone_intrusions": after_zone - before_zone,
            "fall_events":     after_fall - before_fall,
        },
    }


_CAMERA_DEFAULTS = {
    "CAM-08": ("Warehouse Aisle 4", "Warehouse Aisle 4 — Zone B"),
}

def _ensure_camera(camera_id: str, db: Session) -> None:
    """Create a placeholder camera row if one doesn't exist yet."""
    if not db.query(Camera).filter(Camera.id == camera_id).first():
        name, location = _CAMERA_DEFAULTS.get(camera_id, (f"Camera {camera_id}", "Warehouse"))
        db.add(Camera(
            id=camera_id,
            name=name,
            location=location,
            rtsp_url=f"rtsp://localhost/{camera_id}",
            is_active=False,
        ))
        db.commit()


@router.post("/debug-image")
async def debug_image(
    file: UploadFile = File(...),
):
    """Returns raw model detections for debugging — no DB writes."""
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError:
        raise HTTPException(status_code=500, detail="OpenCV/ultralytics not installed")

    content = await file.read()
    arr   = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    ppe_det, _, _ = _get_detectors()
    if ppe_det.model is None:
        raise HTTPException(status_code=500, detail="PPE model not loaded")

    # Run at very low confidence so we see everything the model knows about
    results = ppe_det.model(frame, conf=0.05, verbose=False)[0]
    detections = []
    for box in results.boxes:
        label = results.names[int(box.cls)]
        detections.append({
            "label": label,
            "conf": round(float(box.conf), 3),
            "xyxy": [round(c) for c in box.xyxy[0].tolist()],
        })

    return {
        "model_classes": list(ppe_det.model.names.values()),
        "direct_violation_mode": ppe_det._direct_mode,
        "image_size": {"w": frame.shape[1], "h": frame.shape[0]},
        "detections_at_conf_0.05": detections,
        "active_threshold": ppe_det.model.overrides.get("conf", "N/A"),
    }


@router.post("/analyze-image")
async def analyze_image(
    camera_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    _ensure_camera(camera_id, db)

    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20 MB guard
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _run_image(content, camera_id, db),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Image analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    return result


@router.post("/analyze-video")
async def analyze_video(
    camera_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    _ensure_camera(camera_id, db)

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}")

    content = await file.read()
    if len(content) > 500 * 1024 * 1024:  # 500 MB guard
        raise HTTPException(status_code=413, detail="File too large (max 500 MB)")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _run_video(content, ext, camera_id, db),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Video analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    return result
