from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func

from src.database.connection import get_db
from src.database.models import AttendanceRecord, Student, Session as ClassSession, SpoofAttempt
from src.face_recognition.detector import FaceDetector
from src.face_recognition.recognizer import FaceRecognizer
from src.face_recognition.enrollment import EnrollmentManager
from src.camera.capture import CameraManager

router = APIRouter()

detector = FaceDetector()
recognizer = FaceRecognizer()
enrollment_mgr = EnrollmentManager()


class AttendanceRecordResponse(BaseModel):
    id: int
    student_id: int
    student_name: str
    roll_number: str
    session_id: int
    marked_at: datetime
    confidence: float
    is_spoofed: bool

    class Config:
        from_attributes = True


@router.post("/mark-live")
def mark_live_attendance(session_id: Optional[int] = None, db: DBSession = Depends(get_db)):
    """
    Captures live webcam feed, runs multi-face detection & recognition,
    and marks attendance for recognized students in the given (or latest) session.
    """
    # 1. Resolve active session
    if session_id:
        session_obj = db.query(ClassSession).filter(ClassSession.id == session_id).first()
        if not session_obj:
            raise HTTPException(status_code=404, detail="Specified session does not exist.")
    else:
        # Pick latest active session or create a default session
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

    # 2. Capture a frame from active camera worker
    try:
        from src.main import camera_worker
        frame = camera_worker.get_frame()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera error: {str(e)}")

    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame from active camera.")

    # 3. Detect faces
    detections = detector.detect_faces(frame)
    if not detections:
        return {
            "status": "success",
            "session_id": session_obj.id,
            "faces_detected": 0,
            "marked_students": [],
            "message": "No faces detected in front of the camera."
        }

    # 4. Load known enrolled face embeddings
    known_embeddings = enrollment_mgr.load_known_embeddings(db=db)
    if not known_embeddings:
        return {
            "status": "warning",
            "session_id": session_obj.id,
            "faces_detected": len(detections),
            "marked_students": [],
            "message": "Faces detected, but no students have enrolled their face embeddings yet."
        }

    # 5. Recognize faces and mark attendance
    marked_students = []
    already_marked = []

    for det in detections:
        bbox = det["bbox"]
        # Extract face crop
        x1, y1, x2, y2 = bbox
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        try:
            emb = recognizer.extract_embedding(face_crop)
            matched_id, score = recognizer.identify(emb, known_embeddings)
            
            if matched_id is not None:
                student = db.query(Student).filter(Student.id == matched_id).first()
                if not student:
                    continue

                # Check if already marked for this session
                existing = db.query(AttendanceRecord).filter(
                    AttendanceRecord.student_id == matched_id,
                    AttendanceRecord.session_id == session_obj.id
                ).first()

                if existing:
                    already_marked.append({
                        "student_id": student.id,
                        "name": student.name,
                        "roll_number": student.roll_number,
                        "confidence": round(score, 3)
                    })
                else:
                    record = AttendanceRecord(
                        student_id=student.id,
                        session_id=session_obj.id,
                        confidence=float(score),
                        is_spoofed=False,
                        marked_at=datetime.utcnow()
                    )
                    db.add(record)
                    marked_students.append({
                        "student_id": student.id,
                        "name": student.name,
                        "roll_number": student.roll_number,
                        "confidence": round(score, 3)
                    })
        except Exception:
            continue

    db.commit()

    return {
        "status": "success",
        "session_id": session_obj.id,
        "faces_detected": len(detections),
        "newly_marked": marked_students,
        "already_present": already_marked,
        "message": f"Processed {len(detections)} face(s). Marked {len(marked_students)} student(s)."
    }


@router.get("/")
def list_attendance(session_id: Optional[int] = None, db: DBSession = Depends(get_db)):
    """List attendance records with student details."""
    query = db.query(
        AttendanceRecord.id,
        AttendanceRecord.student_id,
        Student.name.label("student_name"),
        Student.roll_number,
        AttendanceRecord.session_id,
        AttendanceRecord.marked_at,
        AttendanceRecord.confidence,
        AttendanceRecord.is_spoofed
    ).join(Student, AttendanceRecord.student_id == Student.id)

    if session_id:
        query = query.filter(AttendanceRecord.session_id == session_id)

    records = query.order_by(AttendanceRecord.marked_at.desc()).all()
    
    return [
        {
            "id": r.id,
            "student_id": r.student_id,
            "student_name": r.student_name,
            "roll_number": r.roll_number,
            "session_id": r.session_id,
            "marked_at": r.marked_at.isoformat(),
            "confidence": r.confidence,
            "is_spoofed": r.is_spoofed
        }
        for r in records
    ]


@router.get("/stats")
def attendance_stats(db: DBSession = Depends(get_db)):
    """Retrieve aggregate attendance statistics."""
    total_students = db.query(Student).filter(Student.is_active == True).count()
    total_sessions = db.query(ClassSession).count()
    total_attendance = db.query(AttendanceRecord).count()
    spoofs_prevented = db.query(SpoofAttempt).count()

    avg_attendance = 0.0
    if total_sessions > 0 and total_students > 0:
        avg_attendance = round((total_attendance / (total_sessions * total_students)) * 100, 1)

    return {
        "total_enrolled_students": total_students,
        "total_sessions": total_sessions,
        "total_attendance_records": total_attendance,
        "average_attendance_percentage": min(100.0, avg_attendance),
        "spoof_attempts_prevented": spoofs_prevented
    }
