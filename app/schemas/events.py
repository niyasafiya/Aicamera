import json
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, field_validator


class CameraCreate(BaseModel):
    id: str
    name: str
    rtsp_url: str
    location: Optional[str] = None


class CameraOut(BaseModel):
    id: str
    name: str
    rtsp_url: str
    location: Optional[str]
    is_active: bool
    created_at: datetime
    model_config = {"from_attributes": True}


class PPEViolationOut(BaseModel):
    id: int
    camera_id: str
    timestamp: datetime
    missing_ppe: List[str]
    confidence: float
    frame_path: Optional[str]
    acknowledged: bool
    model_config = {"from_attributes": True}

    @field_validator("missing_ppe", mode="before")
    @classmethod
    def parse_missing_ppe(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v or []


class ZoneIntrusionOut(BaseModel):
    id: int
    camera_id: str
    timestamp: datetime
    zone_name: str
    confidence: float
    frame_path: Optional[str]
    acknowledged: bool
    model_config = {"from_attributes": True}


class FallEventOut(BaseModel):
    id: int
    camera_id: str
    timestamp: datetime
    event_type: str
    duration_seconds: Optional[float]
    frame_path: Optional[str]
    resolved: bool
    resolved_at: Optional[datetime]
    model_config = {"from_attributes": True}


class AcknowledgeUpdate(BaseModel):
    acknowledged: bool = True


class FallResolve(BaseModel):
    resolved: bool = True


class ZoneConfig(BaseModel):
    name: str
    polygon: List[List[int]]  # [[x1,y1],[x2,y2],...]
