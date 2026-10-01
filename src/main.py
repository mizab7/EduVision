import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
import cv2
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import get_settings
from src.database.connection import init_db
from src.api.routes import students, attendance, engagement, sessions
from src.face_recognition.detector import FaceDetector
from src.camera.capture import CameraWorker

settings = get_settings()
detector = FaceDetector()
camera_worker = CameraWorker.get_instance(default_index=settings.CAMERA_INDEX)


def generate_camera_frames():
    """Generates continuous MJPEG frames from the background CameraWorker."""
    while True:
        frame = camera_worker.get_frame()
        if frame is None:
            time.sleep(0.04)
            continue

        # Run face detection
        detections = detector.detect_faces(frame)
        annotated_frame = detector.draw_detections(frame, detections)

        # Encode frame as JPEG
        success, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            time.sleep(0.02)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        # Target ~25-30 FPS stream
        time.sleep(0.03)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the database
    init_db()
    yield
    # Shutdown: Clean up camera hardware worker
    camera_worker.release()


app = FastAPI(
    title="EduVision AI API",
    description="Intelligent smart classroom management system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(students.router, prefix="/students", tags=["Students"])
app.include_router(attendance.router, prefix="/attendance", tags=["Attendance"])
app.include_router(engagement.router, prefix="/engagement", tags=["Engagement"])
app.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])


@app.get("/", tags=["UI"])
def serve_portal():
    """Serves the interactive EduVision AI web management portal."""
    portal_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(str(portal_path))


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "ok", "app": "EduVision AI Backend"}


@app.get("/camera/devices", tags=["Camera"])
def list_cameras():
    """Detects and returns all available hardware cameras."""
    devices = CameraWorker.list_available_cameras(max_tested=3)
    status = camera_worker.get_status()
    return {
        "devices": devices,
        "status": status
    }


class CameraConfigPayload(BaseModel):
    index: int
    low_light_boost: bool = False


@app.post("/camera/config", tags=["Camera"])
def configure_camera(config: CameraConfigPayload):
    """Switches active camera index and toggles low-light enhancement filter."""
    camera_worker.switch_camera(config.index, config.low_light_boost)
    return {
        "status": "success",
        "camera_status": camera_worker.get_status(),
        "message": f"Camera switched to index {config.index} (Low-Light: {config.low_light_boost})"
    }


@app.get("/camera/stream", tags=["Camera"])
def video_feed():
    """Streams live MJPEG camera feed with real-time face detection overlays."""
    return StreamingResponse(
        generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """Placeholder for WebSocket endpoint for live classroom data stream."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except Exception as e:
        print(f"WebSocket connection closed: {e}")


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
