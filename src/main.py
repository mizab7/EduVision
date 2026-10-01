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
from src.api.routes import students, attendance, engagement, sessions, ai_assistant, analytics
from src.face_recognition.detector import FaceDetector
from src.anti_spoofing.liveness import AntiSpoofDetector
from src.camera.capture import CameraWorker

from src.engagement.analyzer import EngagementAnalyzer

settings = get_settings()
detector = FaceDetector()
anti_spoof = AntiSpoofDetector(confidence_threshold=0.50)
engagement_analyzer = EngagementAnalyzer()
camera_worker = CameraWorker.get_instance(default_index=settings.CAMERA_INDEX)

# Active visual overlay mode: 'dual', 'engagement', 'attendance'
current_overlay_mode = "dual"


def generate_camera_frames():
    """Generates continuous MJPEG frames with real-time AI computer vision overlays."""
    global current_overlay_mode
    frame_counter = 0

    while True:
        try:
            frame = camera_worker.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            frame_counter += 1
            annotated_frame = frame.copy()

            try:
                detections = detector.detect_faces(frame)

                if current_overlay_mode == "attendance":
                    # Classic attendance & anti-spoofing mode
                    for det in detections:
                        bbox = det["bbox"]
                        liveness = anti_spoof.analyze_liveness(frame, bbox)
                        is_live = liveness["is_live"]

                        if is_live:
                            color = (0, 255, 0)
                            label = f"LIVE ({int(liveness['liveness_score']*100)}%)"
                        else:
                            color = (0, 0, 255)
                            label = f"SPOOF: {liveness['label'].upper()}"

                        cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        for _, pt in det["landmarks"].items():
                            cv2.circle(annotated_frame, pt, 2, color, -1)

                        tag_w = len(label) * 9 + 10
                        cv2.rectangle(annotated_frame, (bbox[0], max(0, bbox[1] - 22)), (bbox[0] + tag_w, bbox[1]), color, -1)
                        cv2.putText(annotated_frame, label, (bbox[0] + 5, max(14, bbox[1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                elif current_overlay_mode in ["engagement", "dual"]:
                    # Run YOLO phone detection every 3 frames for blazing 30+ FPS speed
                    run_phone = (frame_counter % 3 == 0)
                    analysis = engagement_analyzer.analyze_frame(
                        frame, detections, run_phone_detection=run_phone
                    )
                    annotated_frame = engagement_analyzer.annotate_frame(frame, analysis)

                    # In dual mode, also highlight spoof attempts in RED over the engagement box
                    if current_overlay_mode == "dual":
                        for det in detections:
                            bbox = det["bbox"]
                            liveness = anti_spoof.analyze_liveness(frame, bbox)
                            if not liveness["is_live"]:
                                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
                                spoof_tag = f"SPOOF: {liveness['label'].upper()}"
                                tag_w = len(spoof_tag) * 9 + 10
                                cv2.rectangle(annotated_frame, (bbox[0], max(0, bbox[1] - 22)), (bbox[0] + tag_w, bbox[1]), (0, 0, 255), -1)
                                cv2.putText(annotated_frame, spoof_tag, (bbox[0] + 5, max(14, bbox[1] - 6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            except Exception:
                annotated_frame = frame

            # Normalize display stream resolution to 1280x720 HD to prevent browser decoder freezing
            h, w = annotated_frame.shape[:2]
            if (w, h) != (1280, 720):
                annotated_frame = cv2.resize(annotated_frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

            # Encode frame as JPEG
            success, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                time.sleep(0.03)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.03)
        except GeneratorExit:
            break
        except Exception:
            time.sleep(0.05)


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
app.include_router(ai_assistant.router, prefix="/ai-assistant", tags=["AI Teaching Assistant"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics & Predictions"])


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


class OverlayModePayload(BaseModel):
    mode: str  # 'dual', 'engagement', 'attendance'


@app.get("/camera/overlay-mode", tags=["Camera"])
def get_overlay_mode():
    """Returns the currently active AI computer vision overlay mode."""
    global current_overlay_mode
    return {"mode": current_overlay_mode}


@app.post("/camera/overlay-mode", tags=["Camera"])
def set_overlay_mode(payload: OverlayModePayload):
    """Sets the active AI overlay mode: 'dual', 'engagement', or 'attendance'."""
    global current_overlay_mode
    if payload.mode in ["dual", "engagement", "attendance"]:
        current_overlay_mode = payload.mode
    return {"status": "success", "mode": current_overlay_mode}


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
