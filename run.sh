#!/bin/bash
# EduVision AI - One-Click Startup Script

cd "$(dirname "$0")"

echo "============================================="
echo "  🚀 Starting EduVision AI Smart Classroom   "
echo "============================================="

# Ensure virtual environment exists
if [ ! -f "./venv/bin/python" ]; then
    echo "❌ Error: Virtual environment not found in ./venv"
    echo "Please create it using: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Free port 8000 if occupied by a previous instance
PID=$(lsof -ti :8000)
if [ -n "$PID" ]; then
    echo "⚠️ Port 8000 is occupied by process $PID. Freeing port..."
    kill -9 $PID 2>/dev/null
    sleep 1
fi

echo "📦 Initializing AI Models & Cameras..."
echo "🌐 Dashboard:   http://localhost:8000"
echo "📖 API Docs:    http://localhost:8000/docs"
echo "🎥 Stream:      http://localhost:8000/camera/stream"
echo "============================================="
echo "Press Ctrl+C to stop the application."
echo "============================================="

# Automatically open browser after 2 seconds in background
(sleep 2 && open http://localhost:8000) &

# Run FastAPI backend with Uvicorn
exec ./venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
