from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/safety_db"

    PPE_MODEL_PATH: str = "yolov8n.pt"
    POSE_MODEL_PATH: str = "yolov8n-pose.pt"
    PERSON_MODEL_PATH: str = "yolov8n.pt"

    PPE_CONFIDENCE: float = 0.2
    ZONE_CONFIDENCE: float = 0.4
    FALL_CONFIDENCE: float = 0.4

    MOTIONLESS_SECONDS: int = 60
    ALERT_COOLDOWN_SECONDS: int = 30
    FRAME_SKIP: int = 5

    ALERT_IMAGE_DIR: str = "alert_images"

    class Config:
        env_file = ".env"


settings = Settings()
