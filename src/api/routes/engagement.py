"""
Real-time Student Engagement Analysis API routes for EduVision AI.
Provides live metrics, multi-signal breakdowns, and historical logs.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func

from src.database.connection import get_db
from src.database.models import EngagementLog, Student, Session as ClassSession
from src.engagement.analyzer import EngagementAnalyzer
from src.face_recognition.detector import FaceDetector
from src.face_recognition.recognizer import FaceRecognizer
from src.face_recognition.enrollment import EnrollmentManager

router = APIRouter()

detector = FaceDetector()
analyzer = EngagementAnalyzer()
recognizer = FaceRecognizer()
enrollment_mgr = EnrollmentManager()


class EngagementLogResponse(BaseModel):
    id: int
    student_id: Optional[int]
    student_name: Optional[str]
    session_id: Optional[int]
    timestamp: datetime
    attentiveness_score: float
    gaze_score: float
    head_pose_score: float
    blink_rate: float
    is_yawning: bool
    is_drowsy: bool
    phone_detected: bool

    class Config:
        from_attributes = True


@router.get("/live")
def get_live_engagement(session_id: Optional[int] = None, db: DBSession = Depends(get_db)):
    """
    Captures current frame from camera worker and returns live engagement metrics,
    including individual attentiveness scores and Class Engagement Index (CEI).
    """
    from src.main import camera_worker
    frame = camera_worker.get_frame()
    if frame is None:
        return {
            "status": "waiting",
            "class_engagement_index": 0.0,
            "students_detected": 0,
            "students": [],
            "active_alerts": [],
            "message": "Camera stream is initializing..."
        }

    # 1. Detect faces
    faces = detector.detect_faces(frame)

    # 2. Analyze engagement
    analysis = analyzer.analyze_frame(frame, faces, run_phone_detection=True)

    # 3. Identify recognized students
    known_embeddings = enrollment_mgr.load_known_embeddings(db)
    students_output = []

    for s in analysis["students"]:
        face_idx = s["face_idx"]
        bbox = s["bbox"]
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        student_name = f"Student {face_idx + 1}"
        student_id = None

        if crop.size > 0 and known_embeddings:
            try:
                emb = recognizer.extract_embedding(crop)
                matched_id, conf = recognizer.identify(emb, known_embeddings)
                if matched_id is not None:
                    db_student = db.query(Student).filter(Student.id == matched_id).first()
                    if db_student:
                        student_name = db_student.name
                        student_id = db_student.id
            except Exception:
                pass

        students_output.append({
            "student_id": student_id,
            "name": student_name,
            "bbox": bbox,
            "score": s["score"],
            "level": s["level"],
            "posture": s["pose"]["posture_label"],
            "gaze": s["gaze"]["gaze_direction"],
            "is_drowsy": s["blink"]["is_drowsy"],
            "is_yawning": s["yawn"]["is_yawning"],
            "phone_detected": s["phone_detected"],
            "alerts": s["alerts"],
            "breakdown": s["breakdown"]
        })

    return {
        "status": "success",
        "session_id": session_id,
        "class_engagement_index": analysis["class_engagement_index"],
        "students_detected": len(students_output),
        "phones_detected": len(analysis["phones"]),
        "students": students_output,
        "active_alerts": analysis["active_alerts"],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/snapshot")
def save_engagement_snapshot(session_id: Optional[int] = None, db: DBSession = Depends(get_db)):
    """
    Captures live engagement, records an EngagementLog in the database for each student,
    and updates the session's overall Class Engagement Index.
    """
    # Resolve session
    if session_id:
        session_obj = db.query(ClassSession).filter(ClassSession.id == session_id).first()
    else:
        session_obj = db.query(ClassSession).filter(ClassSession.ended_at == None).order_by(ClassSession.started_at.desc()).first()

    if not session_obj:
        session_obj = ClassSession(
            subject="General Classroom Session",
            faculty_name="Faculty Guide",
            classroom="Smart Room 101"
        )
        db.add(session_obj)
        db.commit()
        db.refresh(session_obj)

    from src.main import camera_worker
    frame = camera_worker.get_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame from active camera.")

    faces = detector.detect_faces(frame)
    if not faces:
        return {
            "status": "no_faces",
            "message": "No students detected in camera view to evaluate.",
            "class_engagement_index": 0.0
        }

    analysis = analyzer.analyze_frame(frame, faces, run_phone_detection=True)
    known_embeddings = enrollment_mgr.load_known_embeddings(db)

    logged_count = 0
    now = datetime.utcnow()

    for s in analysis["students"]:
        bbox = s["bbox"]
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        student_id = None
        if crop.size > 0 and known_embeddings:
            try:
                emb = recognizer.extract_embedding(crop)
                matched_id, _ = recognizer.identify(emb, known_embeddings)
                if matched_id:
                    student_id = matched_id
            except Exception:
                pass

        log_entry = EngagementLog(
            student_id=student_id,
            session_id=session_obj.id,
            timestamp=now,
            attentiveness_score=s["score"],
            gaze_score=s["breakdown"]["gaze"],
            head_pose_score=s["breakdown"]["head_pose"],
            blink_rate=float(s["blink"]["blink_rate_per_min"]),
            is_yawning=s["yawn"]["is_yawning"],
            is_drowsy=s["blink"]["is_drowsy"],
            phone_detected=s["phone_detected"]
        )
        db.add(log_entry)
        logged_count += 1

    # Update session CEI
    session_obj.class_engagement_index = analysis["class_engagement_index"]
    db.commit()

    return {
        "status": "success",
        "session_id": session_obj.id,
        "class_engagement_index": analysis["class_engagement_index"],
        "records_logged": logged_count,
        "active_alerts": analysis["active_alerts"]
    }


@router.get("/history")
def get_historical_engagement(session_id: Optional[int] = None, limit: int = 50, db: DBSession = Depends(get_db)):
    """Retrieves recent engagement records from database."""
    query = db.query(
        EngagementLog.id,
        EngagementLog.student_id,
        Student.name.label("student_name"),
        EngagementLog.session_id,
        EngagementLog.timestamp,
        EngagementLog.attentiveness_score,
        EngagementLog.gaze_score,
        EngagementLog.head_pose_score,
        EngagementLog.blink_rate,
        EngagementLog.is_yawning,
        EngagementLog.is_drowsy,
        EngagementLog.phone_detected
    ).outerjoin(Student, EngagementLog.student_id == Student.id)

    if session_id:
        query = query.filter(EngagementLog.session_id == session_id)

    records = query.order_by(EngagementLog.timestamp.desc()).limit(limit).all()

    return [
        {
            "id": r.id,
            "student_id": r.student_id,
            "student_name": r.student_name or "Anonymous / Guest",
            "session_id": r.session_id,
            "timestamp": r.timestamp.isoformat(),
            "attentiveness_score": round(r.attentiveness_score, 1),
            "gaze_score": round(r.gaze_score, 1),
            "head_pose_score": round(r.head_pose_score, 1),
            "blink_rate": r.blink_rate,
            "is_yawning": r.is_yawning,
            "is_drowsy": r.is_drowsy,
            "phone_detected": r.phone_detected
        }
        for r in records
    ]
