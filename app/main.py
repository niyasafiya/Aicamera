import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.database import Base, engine, get_db_sync
from app.models import events  # registers all ORM models before create_all
from app.api.routes import safety as safety_routes
from app.api.routes import video as video_routes
from app.api.routes.safety import set_stream_manager
from app.services.stream_manager import StreamManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    stream_manager = StreamManager()
    set_stream_manager(stream_manager)

    # Auto-start streams for cameras that were active before shutdown
    db = get_db_sync()
    try:
        from app.models.events import Camera
        for cam in db.query(Camera).filter(Camera.is_active == True).all():
            stream_manager.add_camera(cam.id, cam.rtsp_url)
    finally:
        db.close()

    yield

    stream_manager.stop_all()


app = FastAPI(
    title="Sentinel — People & Safety Monitoring API",
    description="FR-P1 · FR-P2 · FR-P3 — PPE, Zone Intrusion, Fall Detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development; restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(safety_routes.router)
app.include_router(video_routes.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/", tags=["meta"])
def frontend():
    return FileResponse("frontend.html")
