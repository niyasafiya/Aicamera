"""
People & Safety endpoints — PPE compliance (hard-hat / vest) and worker
activity (working / on-phone / chatting / resting).

Combines services.ppe_service (person + PPE detection) and
services.activity_service (pose + phone). Mirrors the ANPR/biometric routers:
upload an image or short clip, get back an annotated frame + structured results,
and a row in the safety_log for the event feed.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

import db
from services import activity_service, ppe_service

log = logging.getLogger(__name__)
router = APIRouter()

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_event(kind: str, detail: str, people: int, violations: str, location: str):
    try:
        conn = db.get_conn()
        conn.execute(
            "INSERT INTO safety_log (kind, detail, people, violations, location) "
            "VALUES (?,?,?,?,?)",
            (kind, detail, people, violations, location),
        )
        conn.commit()
        conn.close()
    except Exception:
        log.exception("safety_log write failed")


def _activity_summary(events: List[dict]) -> str:
    counts: dict = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    if not counts:
        return "no people detected"
    order = ["on phone", "chatting", "resting", "working"]
    parts = [f"{counts[k]} {k}" for k in order if k in counts]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Core analysis (runs in a worker thread)
# ---------------------------------------------------------------------------

def _analyze_frame(frame) -> dict:
    """Run PPE + activity on a single BGR frame and build the API response."""
    ppe = ppe_service.get_ppe_detector()
    if not ppe.available:
        raise RuntimeError("Detection models unavailable (ultralytics/model not loaded)")

    analysis = ppe.analyze_frame(frame)
    persons_xyxy = [b["xyxy"] for b in analysis["boxes"]]
    locations = [b["location"] for b in analysis["boxes"]]

    activity = activity_service.get_activity_detector()
    activity_events = activity.classify(frame, persons_xyxy, locations)

    # Merge each person's activity into their PPE box label for the annotation
    act_by_person = {e.get("person"): e["type"]
                     for e in activity_events if "person" in e}
    boxes = analysis["boxes"]
    for i, b in enumerate(boxes):
        act = act_by_person.get(i + 1)
        if act and act != "working":
            b["label"] = f"{b['label']} · {act}"

    annotated = ppe_service.draw_boxes(frame, boxes)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64 = ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
               if ok else None)

    return {
        "status": "complete",
        "people": len(boxes),
        "has_violation": analysis["has_violation"],
        "violations": analysis["violations"],
        "persons": analysis["persons"],
        "activity_events": activity_events,
        "activity_summary": _activity_summary(activity_events),
        "annotated_image": img_b64,
    }


def _run_image(content: bytes) -> dict:
    arr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — unsupported format?")
    return _analyze_frame(frame)


def _run_video(content: bytes, suffix: str) -> dict:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file — unsupported format?")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        max_frames = min(total or int(fps * 30), int(fps * 30))  # analyse ≤ 30 s
        step = max(1, int(fps))                                  # ~1 frame/sec — faster
        MAX_SAMPLES = 15                                         # cap total AI passes

        ppe = ppe_service.get_ppe_detector()
        if not ppe.available:
            raise RuntimeError("Detection models unavailable")

        best_frame = None
        best_count = -1
        any_violation = False
        idx = 0
        sampled = 0
        # Sequential grab/retrieve — far faster than cap.set() seeking on H.264.
        while idx < max_frames and sampled < MAX_SAMPLES:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                fh, fw = frame.shape[:2]
                if fw > 960:                                     # smaller = faster
                    frame = cv2.resize(frame, (960, int(fh * 960 / fw)))
                a = ppe.analyze_frame(frame)
                sampled += 1
                any_violation = any_violation or a["has_violation"]
                if len(a["boxes"]) > best_count:
                    best_count = len(a["boxes"])
                    best_frame = frame.copy()
                    # A frame with people AND a violation is the most informative —
                    # stop early once we have one so scanning stays quick.
                    if a["has_violation"] and best_count > 0:
                        break
            idx += 1
        cap.release()

        if best_frame is None:
            return {"status": "complete", "people": 0, "has_violation": False,
                    "violations": [], "persons": [], "activity_events": [],
                    "activity_summary": "no people detected", "annotated_image": None}

        result = _analyze_frame(best_frame)
        # The clip may contain a violation on a frame other than the sample.
        result["has_violation"] = result["has_violation"] or any_violation
        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/status")
def status():
    ppe = ppe_service.get_ppe_detector()
    return {
        "available": ppe.available,
        "ppe_model_loaded": ppe.model is not None,
        "hat_detection": ppe._has_hat,
        "vest_detection": ppe._has_vest,
    }


@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _IMG_EXT:
        raise HTTPException(400, f"Unsupported image type '{ext}'. Allowed: {', '.join(sorted(_IMG_EXT))}")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 20 MB)")
    try:
        result = await asyncio.to_thread(_run_image, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("Image analysis failed")
        raise HTTPException(500, f"Analysis error: {e}")

    _persist(result)
    return result


@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _VID_EXT:
        raise HTTPException(400, f"Unsupported video type '{ext}'. Allowed: {', '.join(sorted(_VID_EXT))}")
    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(413, "Video too large (max 500 MB)")
    try:
        result = await asyncio.to_thread(_run_video, content, ext)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("Video analysis failed")
        raise HTTPException(500, f"Analysis error: {e}")

    _persist(result)
    return result


def _persist(result: dict) -> None:
    """Write a PPE and/or activity row to the event feed."""
    people = result.get("people", 0)
    if result.get("has_violation"):
        _log_event("ppe", "Missing " + ", ".join(result.get("violations") or ["PPE"]),
                   people, ", ".join(result.get("violations") or []), "")
    elif people:
        _log_event("ppe", "All workers compliant", people, "", "")

    events = result.get("activity_events") or []
    flagged = [e for e in events if e["type"] in ("on phone", "chatting", "resting")]
    if flagged:
        _log_event("activity", result.get("activity_summary", ""), people,
                   "", flagged[0].get("location", ""))


@router.get("/log")
def safety_log(limit: int = Query(30, ge=1, le=200)):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id, timestamp, kind, detail, people, violations, location "
        "FROM safety_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.delete("/log/{log_id}", status_code=204)
def delete_safety_log(log_id: int):
    """Remove a single PPE / activity event from the feed."""
    conn = db.get_conn()
    conn.execute("DELETE FROM safety_log WHERE id=?", (log_id,))
    conn.commit()
    conn.close()


@router.delete("/log", status_code=204)
def clear_safety_log(kind: Optional[str] = Query(None)):
    """Clear the whole feed, or only 'ppe' / 'activity' rows when kind is given."""
    conn = db.get_conn()
    if kind in ("ppe", "activity"):
        conn.execute("DELETE FROM safety_log WHERE kind=?", (kind,))
    else:
        conn.execute("DELETE FROM safety_log")
    conn.commit()
    conn.close()
