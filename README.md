# EduVision AI 🎓👁️

> **An Intelligent Smart Classroom Management System Using Multi-Face Recognition, Anti-Spoofing, Real-Time Student Engagement Analysis, and AI Teaching Assistant**

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React.js-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/ML-MediaPipe-007FFF.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![InsightFace](https://img.shields.io/badge/Face_Recognition-InsightFace-orange.svg)](https://github.com/deepinsight/insightface)
[![YOLOv8](https://img.shields.io/badge/Object_Detection-YOLOv8-FF6F00.svg)](https://ultralytics.com/)
[![Google Gemini](https://img.shields.io/badge/GenAI-Google%20Gemini%20API-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
  - [1. Multi-Face Recognition](#1-multi-face-recognition)
  - [2. Anti-Spoofing & Liveness Detection](#2-anti-spoofing--liveness-detection)
  - [3. Multimodal Engagement Analysis (6 Signals)](#3-multimodal-engagement-analysis-6-signals)
  - [4. AI Teaching Assistant](#4-ai-teaching-assistant)
  - [5. Real-Time Analytics Dashboard](#5-real-time-analytics-dashboard)
  - [6. Predictive Analytics](#6-predictive-analytics)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
- [Running the Application](#-running-the-application)
- [API Endpoints](#-api-endpoints)
- [Team](#-team)
- [License](#-license)

---

## 📖 Overview

**EduVision AI** is an end-to-end intelligent classroom monitoring and educational enhancement platform designed for modern academic institutions. Traditional classroom management suffers from manual attendance inefficiencies, unmonitored proxy attendance, delayed identification of disengaged or struggling students, and a lack of actionable pedagogical feedback for educators.

EduVision AI bridges this gap by harmonizing state-of-the-art **Computer Vision**, **Edge Deep Learning**, and **Generative AI**:
- **Automated Multi-Student Attendance:** Concurrently registers and marks attendance for multiple students in a single video frame with biometric precision.
- **Robust Anti-Spoofing:** Enforces liveness checks to stop digital playback, 2D printed photographs, and silicone mask impersonation.
- **Multimodal Engagement Assessment:** Analyzes six distinct non-invasive behavioral signals in real time to compute a continuous engagement score for every student.
- **Generative AI Classroom Co-Pilot:** Integrates Google Gemini API to produce instant lecture summaries, student comprehension insights, and automated alert interventions for teachers.

---

## ✨ Key Features

### 1. Multi-Face Recognition
- **Concurrent Detection & Recognition:** Identifies multiple students simultaneously in crowded classroom frames using InsightFace and SCRFD/RetinaFace detectors.
- **High-Dimensional Embeddings:** Extracts 512-dimensional facial feature vectors for rotation- and scale-invariant verification.
- **Sub-Millisecond Vector Search:** Employs **FAISS** (Facebook AI Similarity Search) indexing to match student vectors instantly across thousands of enrolled records.
- **Automated Attendance Logging:** Records timestamps, confidence scores, and camera metadata with exportable audit logs.

### 2. Anti-Spoofing & Liveness Detection
- **Multi-Factor Liveness:** Blends active and passive liveness verification to neutralize presentation attacks (PAD).
- **Texture & Frequency Analysis:** Analyzes micro-texture patterns (Fourier / LBP analysis) to detect screen moiré and paper grain reflections.
- **Blink & Micro-Movement Verification:** Monitors natural involuntary blink rhythms and head micro-shifts to confirm live human subjects.

### 3. Multimodal Engagement Analysis (6 Signals)
EduVision AI extracts **six complementary behavioral signals** to compute an objective, real-time engagement score:

| Signal | Metric / Technique | Description |
| :--- | :--- | :--- |
| 👁️ **1. Gaze Tracking** | Pupil/Iris Center Landmark Estimation | Tracks whether student focus is on the board, instructional screen, or off-target |
| 🔄 **2. Head Pose Estimation** | 3D Euler Angles (Yaw, Pitch, Roll) via PnP | Flags turned heads, looking down, sleeping posture, or peer conversations |
| ⚡ **3. Blink Rate Detection** | Eye Aspect Ratio (EAR) Thresholding | Measures blink frequency and prolonged eye closure indicating drowsiness |
| 🥱 **4. Yawn Detection** | Mouth Aspect Ratio (MAR) Dynamics | Detects open-mouth yawning frequency to evaluate mental fatigue |
| 📱 **5. Phone & Device Detection** | YOLOv8 Custom Object Detection | Identifies unauthorized smartphone or tablet usage within the student field |
| 📊 **6. Engagement Scorer** | Weighted Multi-Signal Fusion Engine | Computes composite index (0–100%) with rolling temporal smoothing |

### 4. AI Teaching Assistant
- **Google Gemini API Integration:** Generates lecture context-aware summaries and highlights moments of high confusion or low collective engagement.
- **Automated Question Generation:** Crafts formative assessment questions dynamically tailored to topics covered when attention dips.
- **Intelligent Teacher Alerts:** Sends immediate, gentle nudges to the teacher's dashboard (e.g., *"Row 3 attention dropped by 35% during the last 5 minutes"*).

### 5. Real-Time Analytics Dashboard
- **Live Classroom Heatmaps:** Visual grid representation of student engagement levels across physical seating arrangements.
- **Temporal Attention Trends:** Real-time line graphs showing class-wide focus throughout the lecture timeline.
- **Student Profile View:** Individual profiles showing historical attendance rates, focus trends, and participation insights.

### 6. Predictive Analytics
- **Early-Warning Engine:** Employs regression and time-series models to correlate classroom engagement patterns with academic grades.
- **At-Risk Identification:** Automatically flags students at risk of falling behind before major examinations occur.

---

## 🛠️ Tech Stack

| Domain | Technology / Library | Role & Functionality |
| :--- | :--- | :--- |
| **Frontend** | **React.js** (v18+) | Interactive responsive single-page instructor dashboard |
| | **Tailwind CSS** | Modern UI styling, dark/light mode, and responsive layout |
| | **Recharts / Chart.js** | Live telemetry, engagement charts, and attendance graphs |
| | **WebSockets API** | Low-latency bi-directional streaming of metrics and video overlays |
| **Backend** | **FastAPI** | High-performance asynchronous Python REST & WebSocket framework |
| | **Uvicorn** | Lightning-fast ASGI production web server |
| | **Pydantic (v2)** | Data validation and type-safe schema enforcement |
| **ML / Computer Vision** | **OpenCV** (cv2) | Frame capture, video preprocessing, color spaces, drawing overlays |
| | **MediaPipe** | 468-point Face Mesh, iris landmark tracking, facial geometry |
| | **InsightFace** | SOTA face detection, alignment, and 512-d feature extraction |
| | **YOLOv8 (Ultralytics)** | High-fps object detection for mobile phones and digital devices |
| **Database & Vector Index**| **SQLite / PostgreSQL** | Relational storage for student profiles, courses, and attendance |
| | **SQLAlchemy / Alembic** | Python ORM and schema migration manager |
| | **FAISS** | Fast vector similarity search for face embedding retrieval |
| **Generative AI** | **Google Gemini API** | Multi-modal reasoning, classroom summaries, and smart alerts |

---

## 📁 Project Structure

```
eduvision-ai/
├── config/
│   └── settings.py              # Application settings, environment loader & constants
├── src/
│   ├── main.py                  # FastAPI application entrypoint & lifecycle management
│   ├── camera/
│   │   └── capture.py           # Video capture streams (webcam, RTSP, video file playback)
│   ├── face_recognition/
│   │   ├── detector.py          # Face detection & bounding box extraction
│   │   ├── recognizer.py        # 512-d embedding extraction & FAISS similarity matching
│   │   └── enrollment.py        # Student enrollment pipeline & vector database indexing
│   ├── anti_spoofing/
│   │   └── liveness.py          # Anti-spoofing checks (texture, blink, depth, reflection)
│   ├── engagement/
│   │   ├── gaze_tracker.py      # Pupil & iris vector estimation relative to screen/board
│   │   ├── head_pose.py         # 3D head pose estimation (Yaw, Pitch, Roll angles)
│   │   ├── blink_detector.py    # Eye Aspect Ratio (EAR) computation & blink frequency
│   │   ├── yawn_detector.py     # Mouth Aspect Ratio (MAR) computation & yawn detection
│   │   ├── phone_detector.py    # YOLOv8 smartphone and prohibited device detector
│   │   └── scorer.py            # Multimodal temporal scoring & engagement fusion
│   ├── ai_assistant/
│   │   ├── assistant.py         # Google Gemini integration for summaries & query answering
│   │   └── alerts.py            # Rule-based & AI-driven teacher alert generation
│   ├── analytics/
│   │   └── predictor.py         # Predictive model for student performance & risk analysis
│   ├── database/
│   │   ├── models.py            # SQLAlchemy database entities (Student, Attendance, Session)
│   │   └── connection.py        # Database session management & engine initialization
│   └── api/
│       ├── routes/              # REST API endpoint routers (auth, students, attendance, etc.)
│       └── websocket/           # WebSocket handlers for live telemetry & video frame feeds
├── frontend/                    # React.js web dashboard source code
├── models/                      # Pretrained weights (YOLOv8, InsightFace models)
├── data/                        # Local SQLite database, FAISS indices & sample datasets
├── tests/                       # Unit & integration test suites
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python:** `3.10` or higher
- **Package Manager:** `pip`
- **Node.js & npm:** `Node.js 18+` (for running the React frontend)
- **Camera Device:** Integrated webcam, external USB camera, or an IP/RTSP network camera feed
- **API Key:** Google Gemini API Key ([Get one here](https://aistudio.google.com/))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/prar.git
   cd prar
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS / Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   ```bash
   cp .env.example .env
   ```

### Environment Configuration

Configure the `.env` file with your specific credentials:

```ini
# Server Settings
APP_NAME="EduVision AI"
ENVIRONMENT="development"
HOST="0.0.0.0"
PORT=8000
DEBUG=True

# Database Configuration
DATABASE_URL="sqlite:///./data/eduvision.db"
# For PostgreSQL: postgresql+psycopg2://user:password@localhost:5432/eduvision

# Google Gemini API
GEMINI_API_KEY="your-google-gemini-api-key-here"
GEMINI_MODEL="gemini-1.5-flash"

# Camera Settings
CAMERA_INDEX=0
# Or RTSP stream: CAMERA_RTSP_URL="rtsp://admin:password@192.168.1.50:554/stream1"
FRAME_WIDTH=1280
FRAME_HEIGHT=720
FPS_TARGET=30

# Detection & Model Thresholds
FACE_RECOGNITION_THRESHOLD=0.60
LIVENESS_CONFIDENCE_THRESHOLD=0.75
ENGAGEMENT_ALERT_THRESHOLD=45.0
YOLO_PHONE_CONFIDENCE=0.45
```

---

## 💻 Running the Application

### 1. Camera Diagnostic Test
Verify your video capture hardware and OpenCV installation:
```bash
python -m src.camera.capture
```
*(Press `q` on the preview window to exit)*

### 2. Start the Backend API Server
Launch the FastAPI application using the module entrypoint:
```bash
python -m src.main
```
Or directly via Uvicorn with hot-reload enabled:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Interactive API Documentation
Once the server is running, explore and test the endpoints interactively:
- **Interactive Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc UI:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 4. Start the Frontend Dashboard (Optional)
Navigate to the frontend directory and launch the development server:
```bash
cd frontend
npm install
npm start
```
The dashboard will open automatically at [http://localhost:3000](http://localhost:3000).

---

## 🔌 API Endpoints

| HTTP Method | Endpoint | Description | Tag / Area |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health check and uptime diagnostic | System |
| `POST` | `/api/v1/enrollment/register` | Register new student biometric face profile & embeddings | Enrollment |
| `POST` | `/api/v1/attendance/verify` | Perform multi-face recognition & liveness on image frame | Attendance |
| `GET` | `/api/v1/attendance/records` | Query historical attendance logs with date/class filters | Attendance |
| `POST` | `/api/v1/engagement/analyze-frame` | Analyze single frame for 6 multimodal engagement metrics | Engagement |
| `GET` | `/api/v1/engagement/session/{id}`| Retrieve aggregated engagement telemetry for class session | Engagement |
| `POST` | `/api/v1/assistant/chat` | Send pedagogical query to Gemini AI Teaching Assistant | AI Assistant |
| `POST` | `/api/v1/assistant/session-summary`| Request Gemini-generated classroom summary and suggestions | AI Assistant |
| `GET` | `/api/v1/analytics/at-risk` | Predict list of students with declining engagement patterns | Analytics |
| `WS` | `/ws/live-stream` | Real-time WebSocket stream for video frames & bounding overlays | Streaming |
| `WS` | `/ws/alerts` | Real-time WebSocket notification channel for teacher alerts | Telemetry |

---

## 👥 Team

Developed by **B.Tech Artificial Intelligence & Machine Learning** students at **College of Engineering and Technology, Payyanur**:

| Name | Role / Area | GitHub / Profile |
| :--- | :--- | :--- |
| 🧑‍💻 **Adnan K** | AI & Machine Learning | [@adnank](https://github.com/) |
| 🧑‍💻 **Zensher Zahir** | Computer Vision & Backend | [@zensherzahir](https://github.com/) |
| 👩‍💻 **Prarthana** | Deep Learning & Engagement Analytics | [@prarthana](https://github.com/) |
| 👩‍💻 **Noofa** | Full-Stack & UI/UX Integration | [@noofa](https://github.com/) |

### 🎓 Project Guidance
- **Faculty Guide:** **Priya T E**
- **Department:** Department of Artificial Intelligence & Machine Learning
- **Institution:** College of Engineering and Technology, Payyanur

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Adnan K, Zensher Zahir, Prarthana, Noofa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  Made with ❤️ by the <b>EduVision AI Team</b> · College of Engineering and Technology, Payyanur
</p>
