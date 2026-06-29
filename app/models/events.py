from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from datetime import datetime
from app.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    rtsp_url = Column(String, nullable=False)
    location = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PPEViolation(Base):
    __tablename__ = "ppe_violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    missing_ppe = Column(String)  # JSON array: ["hard_hat", "safety_vest"]
    confidence = Column(Float)
    frame_path = Column(String)
    acknowledged = Column(Boolean, default=False)


class ZoneIntrusion(Base):
    __tablename__ = "zone_intrusions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    zone_name = Column(String, nullable=False)
    confidence = Column(Float)
    frame_path = Column(String)
    acknowledged = Column(Boolean, default=False)


class FallEvent(Base):
    __tablename__ = "fall_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String)  # "fall" | "motionless"
    duration_seconds = Column(Float, nullable=True)
    frame_path = Column(String)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
