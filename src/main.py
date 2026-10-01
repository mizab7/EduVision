import os
import threading
from pathlib import Path
from contextlib import asynccontextmanager
import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import get_settings
from src.database.connection import init_db
from src.api.routes import students, attendance, engagement, sessions
from src.face_recognition.detector import FaceDetector
from src.camera.capture import CameraManager, enhance_low_light

settings = get_settings()
detector = FaceDetector()

# Global camera state
camera_lock = threading.Lock()
active_camera_index = settings.CAMERA_INDEX
active_low_light_boost = False
_camera_cap = None


def get_camera_stream():
    """Gets or initializes the active cv2.VideoCapture with HD resolution."""
    global _camera_cap, active_camera_index
    with camera_lock:
        if _camera_cap is None or not _camera_cap.isOpened():
            _camera_cap = cv2.VideoCapture(active_camera_index)
            # Request HD resolution
            _camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            _camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return _camera_cap


def set_camera_device(index: int, low_light: bool = False):
    """Safely switches camera hardware device and toggles low-light filter."""
    global _camera_cap, active_camera_index, active_low_light_boost
    with camera_lock:
        active_camera_index = index
        active_low_light_boost = low_light
        if _camera_cap is not None and _camera_cap.isOpened():
            _camera_cap.release()
        _camera_cap = cv2.VideoCapture(active_camera_index)
        _camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        _camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def generate_camera_frames():
    """Generates continuous MJPEG frames with optional low-light boost and face detection."""
    global active_low_light_boost
    while True:
        cap = get_camera_stream()
        if cap is None or not cap.isOpened():
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Low-light enhancement (CLAHE)
        if active_low_light_boost:
            frame = enhance_low_light(frame, clip_limit=3.0)

        # Run face detection
        detections = detector.detect_faces(frame)
        annotated_frame = detector.draw_detections(frame, detections)

        # Encode frame as JPEG
        success, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the database
    init_db()
    yield
    # Shutdown: Clean up camera
    global _camera_cap
    with camera_lock:
        if _camera_cap is not None and _camera_cap.isOpened():
            _camera_cap.release()
            _camera_cap = None


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
    devices = CameraManager.list_available_cameras(max_tested=3)
    return {
        "devices": devices,
        "active_index": active_camera_index,
        "low_light_boost": active_low_light_boost
    }


class CameraConfigPayload(BaseModel):
    index: int
    low_light_boost: bool = False


@app.post("/camera/config", tags=["Camera"])
def configure_camera(config: CameraConfigPayload):
    """Switches active camera index and toggles low-light enhancement filter."""
    set_camera_device(config.index, config.low_light_boost)
    return {
        "status": "success",
        "active_index": active_camera_index,
        "low_light_boost": active_low_light_boost,
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
