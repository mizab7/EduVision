import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config.settings import get_settings
from src.database.connection import init_db
from src.api.routes import students, attendance, engagement, sessions

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the database
    init_db()
    yield
    # Shutdown: Clean up resources if needed

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


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "ok", "app": "EduVision AI Backend"}


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
