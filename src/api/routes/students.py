from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session
import cv2
import numpy as np

from src.database.connection import get_db
from src.database.models import Student, FaceEmbedding
from src.face_recognition.enrollment import EnrollmentManager
from src.camera.capture import CameraManager

router = APIRouter()
enrollment_mgr = EnrollmentManager()


class StudentCreate(BaseModel):
    name: str
    roll_number: str
    department: str
    semester: int


class StudentResponse(BaseModel):
    id: int
    name: str
    roll_number: str
    department: str
    semester: int
    enrolled_at: datetime
    is_active: bool
    is_face_enrolled: Optional[bool] = False

    class Config:
        from_attributes = True


@router.post("/", response_model=StudentResponse)
def create_student(student: StudentCreate, db: Session = Depends(get_db)):
    """Create a new student in the database."""
    existing = db.query(Student).filter(Student.roll_number == student.roll_number).first()
    if existing:
        raise HTTPException(status_code=400, detail="Student with this roll number already exists.")

    new_student = Student(
        name=student.name,
        roll_number=student.roll_number,
        department=student.department,
        semester=student.semester,
        is_active=True
    )
    db.add(new_student)
    db.commit()
    db.refresh(new_student)
    return new_student


@router.get("/", response_model=List[StudentResponse])
def list_students(db: Session = Depends(get_db)):
    """List all registered students."""
    students = db.query(Student).filter(Student.is_active == True).all()
    # Check face enrollment status
    enrolled_ids = {fe.student_id for fe in db.query(FaceEmbedding.student_id).all()}
    
    results = []
    for s in students:
        resp = StudentResponse(
            id=s.id,
            name=s.name,
            roll_number=s.roll_number,
            department=s.department,
            semester=s.semester,
            enrolled_at=s.enrolled_at,
            is_active=s.is_active,
            is_face_enrolled=(s.id in enrolled_ids)
        )
        results.append(resp)
    return results


@router.get("/{student_id}", response_model=StudentResponse)
def get_student(student_id: int, db: Session = Depends(get_db)):
    """Retrieve a student by ID."""
    student = db.query(Student).filter(Student.id == student_id, Student.is_active == True).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    
    is_enrolled = db.query(FaceEmbedding).filter(FaceEmbedding.student_id == student.id).first() is not None
    return StudentResponse(
        id=student.id,
        name=student.name,
        roll_number=student.roll_number,
        department=student.department,
        semester=student.semester,
        enrolled_at=student.enrolled_at,
        is_active=student.is_active,
        is_face_enrolled=is_enrolled
    )


@router.post("/{student_id}/enroll-upload")
async def enroll_student_image(student_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a face photo to generate encrypted embeddings and enroll the student."""
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        enrollment_mgr.enroll_from_images(student_id, [img], db=db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Enrollment failed: {str(e)}")

    return {"status": "success", "message": f"Successfully enrolled face for {student.name}"}


@router.post("/{student_id}/enroll-webcam")
def enroll_student_webcam(student_id: int, db: Session = Depends(get_db)):
    """Captures 3 frames from the live webcam to enroll student face embeddings."""
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    frames = []
    try:
        from src.main import camera_worker
        import time

        for _ in range(6):
            f = camera_worker.get_frame()
            if f is not None:
                frames.append(f)
            time.sleep(0.08)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to access camera: {str(e)}")

    if not frames:
        raise HTTPException(status_code=500, detail="Failed to capture frames from active camera.")

    try:
        enrollment_mgr.enroll_from_images(student_id, frames, db=db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Enrollment failed: {str(e)}")

    return {
        "status": "success",
        "message": f"Successfully enrolled {student.name} via live camera capture ({len(frames)} frames)."
    }


@router.delete("/{student_id}")
def delete_student(student_id: int, db: Session = Depends(get_db)):
    """Soft delete a student."""
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    student.is_active = False
    db.commit()
    return {"message": "Student soft deleted successfully"}
