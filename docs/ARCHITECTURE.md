# EduVision AI — System Architecture & Technical Specifications

This document outlines the detailed architectural blueprints, algorithmic workflows, and pipeline designs for **EduVision AI**.

---

## 🏛️ High-Level System Architecture

```
                      ┌─────────────────────────────────────────┐
                      │          Video Ingestion Layer          │
                      │  (Webcam / RTSP / Network Video Stream) │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │         Preprocessing & Pipeline        │
                      │   (Frame Resizing, Color Conversion)    │
                      └──────────────┬──────────────────┬───────┘
                                     │                  │
                ┌────────────────────┴───┐          ┌───┴────────────────────┐
                ▼                        ▼          ▼                        ▼
     ┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
     │ Face Detection &     │ │  Anti-Spoofing   │ │ Multimodal           │
     │ Recognition          │ │  Liveness Engine │ │ Engagement Engine    │
     │ - RetinaFace/SCRFD   │ │ - Texture/Moiré  │ │ 1. Gaze Tracker      │
     │ - InsightFace 512-d  │ │ - Blink Rhythm   │ │ 2. Head Pose (PnP)   │
     │ - FAISS Similarity   │ │ - 3D Surface Cue │ │ 3. Blink Rate (EAR)  │
     └──────────┬───────────┘ └─────────┬────────┘ │ 4. Yawn (MAR)        │
                │                       │          │ 5. YOLOv8 Phone Det  │
                └───────────┬───────────┘          │ 6. Fusion Scorer     │
                            ▼                      └──────────┬───────────┘
                ┌───────────────────────┐                     │
                │ Attendance Validator  │                     │
                └───────────┬───────────┘                     │
                            │                                 │
                            └────────────────┬────────────────┘
                                             │
                                             ▼
                      ┌─────────────────────────────────────────┐
                      │    FastAPI Central Application Server   │
                      │    - Session Manager                    │
                      │    - WebSocket Event Broadcaster        │
                      │    - Historical Analytics Storage       │
                      └──────────────┬──────────────────┬───────┘
                                     │                  │
                ┌────────────────────┴───┐          ┌───┴────────────────────┐
                ▼                        ▼          ▼                        ▼
     ┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
     │ Relational Database  │ │ Vector Store     │ │ Generative AI        │
     │ (SQLite/PostgreSQL)  │ │ (FAISS Index)    │ │ Assistant            │
     │ - Attendance Records │ │ - 512-d Face     │ │ - Google Gemini 1.5  │
     │ - Engagement History │ │   Embeddings     │ │ - Real-Time Alerts   │
     └──────────────────────┘ └──────────────────┘ └──────────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────────────────┐
                      │       React.js Instructor UI            │
                      │  - Live Video Overlay & Heatmaps        │
                      │  - Attention Timeline & Metrics         │
                      │  - Teacher Actionable Alerts            │
                      └─────────────────────────────────────────┘
```

---

## 🔍 Core Subsystems & Algorithms

### 1. Face Recognition & Vector Matching
1. **Detection:** Detect faces using RetinaFace or SCRFD at low inference latency.
2. **Alignment:** Perform affine transformation using 5 facial keypoints (eyes, nose tip, mouth corners).
3. **Embedding:** Extract a unit-normalized $512$-dimensional vector $\mathbf{v} \in \mathbb{R}^{512}$ via InsightFace (ArcFace backbone).
4. **Vector Retrieval:** Index vectors into a FAISS `IndexFlatIP` (inner product for cosine similarity). Matches with similarity $\ge \tau_{\text{face}}$ (default $0.60$) are registered.

### 2. Anti-Spoofing & Liveness Verification
Combines multi-modal passive checks to prevent fraudulent attendance:
- **Fourier Spectrum Analysis:** Detects high-frequency print artifacts and moiré patterns from mobile/LCD screens.
- **Micro-Blink Verification:** Confirms physiological dynamic eye closure over a sliding $3$-second window.
- **Surface Gradient:** Measures geometric depth cues using MediaPipe dense mesh to confirm a 3D convex face surface rather than a flat planar surface.

### 3. Multimodal Engagement Analysis (6 Signals)

The total engagement score $E_t \in [0, 100]$ at time $t$ fuses 6 real-time signals:

$$E_t = w_{\text{gaze}} S_{\text{gaze}} + w_{\text{pose}} S_{\text{pose}} + w_{\text{blink}} S_{\text{blink}} + w_{\text{yawn}} S_{\text{yawn}} + w_{\text{phone}} S_{\text{phone}} + w_{\text{history}} E_{t-1}$$

1. **Gaze Tracking ($S_{\text{gaze}}$):**
   - MediaPipe Iris landmarks identify pupil offset relative to eye corners.
   - Outputs gaze deviation angle from the central teaching focal zone.
2. **Head Pose Estimation ($S_{\text{pose}}$):**
   - Uses Perspective-n-Point (`cv2.solvePnP`) using 6 canonical 3D facial landmarks.
   - Calculates Euler angles: Yaw ($\psi$), Pitch ($\theta$), Roll ($\phi$).
   - Flags down-turned head or lateral deviation exceeding $25^{\circ}$.
3. **Blink Rate / Drowsiness ($S_{\text{blink}}$):**
   - Evaluates Eye Aspect Ratio (EAR):
     $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$
   - Sustained low EAR ($\text{EAR} < 0.21$ for $>20$ frames) indicates micro-sleep or heavy fatigue.
4. **Yawn Detection ($S_{\text{yawn}}$):**
   - Evaluates Mouth Aspect Ratio (MAR):
     $$\text{MAR} = \frac{\|m_2 - m_8\| + \|m_3 - m_7\| + \|m_4 - m_6\|}{2 \|m_1 - m_5\|}$$
   - Sustained high MAR indicates yawning and mental fatigue.
5. **Unauthorized Device Detection ($S_{\text{phone}}$):**
   - YOLOv8 detects bounding boxes labeled `cell phone`, `laptop`, or `tablet` near students.
   - Imposes immediate distraction penalty when detected during active instruction.
6. **Composite Temporal Fusion ($S_{\text{fusion}}$):**
   - Exponential moving average (EMA) smooths noisy frame-by-frame fluctuations.

### 4. AI Teaching Assistant (Google Gemini 1.5)
- Summarizes rolling 10-minute engagement statistics into human-readable insights.
- Detects collective attention drop across rows and recommends pedagogical pacing changes.
- Generates interactive quiz questions targeted at topics where collective confusion was detected.
