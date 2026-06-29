from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.events import Camera, FallEvent, PPEViolation, ZoneIntrusion
from app.schemas.events import (
    AcknowledgeUpdate,
    CameraCreate,
    CameraOut,
    FallEventOut,
    FallResolve,
    PPEViolationOut,
    ZoneConfig,
    ZoneIntrusionOut,
)

router = APIRouter(prefix="/api/safety", tags=["safety"])

_stream_manager = None


def set_stream_manager(sm) -> None:
    global _stream_manager
    _stream_manager = sm


# ── Cameras ───────────────────────────────────────────────────────────────────

@router.post("/cameras", response_model=CameraOut, status_code=201)
def add_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    if db.query(Camera).filter(Camera.id == payload.id).first():
        raise HTTPException(status_code=409, detail="Camera ID already exists")
    camera = Camera(**payload.model_dump())
    db.add(camera)
    db.commit()
    db.refresh(camera)
    if _stream_manager:
        _stream_manager.add_camera(camera.id, camera.rtsp_url)
    return camera


@router.get("/cameras", response_model=list[CameraOut])
def list_cameras(db: Session = Depends(get_db)):
    return db.query(Camera).filter(Camera.is_active == True).all()


@router.delete("/cameras/{camera_id}", status_code=204)
def remove_camera(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera.is_active = False
    db.commit()
    if _stream_manager:
        _stream_manager.remove_camera(camera_id)


# ── Restricted Zone Configuration ─────────────────────────────────────────────

@router.post("/cameras/{camera_id}/zones")
def set_zones(camera_id: str, zones: list[ZoneConfig], db: Session = Depends(get_db)):
    if not db.query(Camera).filter(Camera.id == camera_id).first():
        raise HTTPException(status_code=404, detail="Camera not found")
    if _stream_manager:
        _stream_manager.zone_monitor.set_zones(
            camera_id,
            [z.model_dump() for z in zones],
        )
    return {"camera_id": camera_id, "zones_set": len(zones)}


# ── PPE Violations (FR-P1) ────────────────────────────────────────────────────

@router.get("/violations/ppe", response_model=list[PPEViolationOut])
def get_ppe_violations(
    camera_id: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    q = db.query(PPEViolation)
    if camera_id:
        q = q.filter(PPEViolation.camera_id == camera_id)
    if acknowledged is not None:
        q = q.filter(PPEViolation.acknowledged == acknowledged)
    return q.order_by(PPEViolation.timestamp.desc()).limit(limit).all()


@router.patch("/violations/ppe/{violation_id}", response_model=PPEViolationOut)
def acknowledge_ppe(
    violation_id: int, payload: AcknowledgeUpdate, db: Session = Depends(get_db)
):
    v = db.query(PPEViolation).filter(PPEViolation.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="Violation not found")
    v.acknowledged = payload.acknowledged
    db.commit()
    db.refresh(v)
    return v


# ── Zone Intrusions (FR-P2) ───────────────────────────────────────────────────

@router.get("/violations/zone", response_model=list[ZoneIntrusionOut])
def get_zone_intrusions(
    camera_id: Optional[str] = Query(None),
    zone_name: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    q = db.query(ZoneIntrusion)
    if camera_id:
        q = q.filter(ZoneIntrusion.camera_id == camera_id)
    if zone_name:
        q = q.filter(ZoneIntrusion.zone_name == zone_name)
    if acknowledged is not None:
        q = q.filter(ZoneIntrusion.acknowledged == acknowledged)
    return q.order_by(ZoneIntrusion.timestamp.desc()).limit(limit).all()


@router.patch("/violations/zone/{intrusion_id}", response_model=ZoneIntrusionOut)
def acknowledge_zone(
    intrusion_id: int, payload: AcknowledgeUpdate, db: Session = Depends(get_db)
):
    v = db.query(ZoneIntrusion).filter(ZoneIntrusion.id == intrusion_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="Intrusion not found")
    v.acknowledged = payload.acknowledged
    db.commit()
    db.refresh(v)
    return v


# ── Fall / Motionless Events (FR-P3) ──────────────────────────────────────────

@router.get("/alerts/fall", response_model=list[FallEventOut])
def get_fall_events(
    camera_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None, description="fall | motionless"),
    resolved: Optional[bool] = Query(None),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    q = db.query(FallEvent)
    if camera_id:
        q = q.filter(FallEvent.camera_id == camera_id)
    if event_type:
        q = q.filter(FallEvent.event_type == event_type)
    if resolved is not None:
        q = q.filter(FallEvent.resolved == resolved)
    return q.order_by(FallEvent.timestamp.desc()).limit(limit).all()


@router.patch("/alerts/fall/{event_id}", response_model=FallEventOut)
def resolve_fall_event(
    event_id: int, payload: FallResolve, db: Session = Depends(get_db)
):
    e = db.query(FallEvent).filter(FallEvent.id == event_id).first()
    if not e:
        raise HTTPException(status_code=404, detail="Fall event not found")
    e.resolved = payload.resolved
    e.resolved_at = datetime.utcnow() if payload.resolved else None
    db.commit()
    db.refresh(e)
    return e
