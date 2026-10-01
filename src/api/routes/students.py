from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.connection import get_db

router = APIRouter()


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

    class Config:
        from_attributes = True


@router.post("/", response_model=StudentResponse)
def create_student(student: StudentCreate, db: Session = Depends(get_db)):
    """Create a new student."""
    # Placeholder implementation
    return {
        "id": 1,
        "name": student.name,
        "roll_number": student.roll_number,
        "department": student.department,
        "semester": student.semester,
        "enrolled_at": datetime.utcnow(),
        "is_active": True
    }


@router.get("/", response_model=List[StudentResponse])
def list_students(db: Session = Depends(get_db)):
    """List all students."""
    # Placeholder implementation
    return []


@router.get("/{student_id}", response_model=StudentResponse)
def get_student(student_id: int, db: Session = Depends(get_db)):
    """Retrieve a student by ID."""
    # Placeholder implementation
    return {
        "id": student_id,
        "name": "John Doe",
        "roll_number": "CS101",
        "department": "Computer Science",
        "semester": 5,
        "enrolled_at": datetime.utcnow(),
        "is_active": True
    }


@router.put("/{student_id}", response_model=StudentResponse)
def update_student(student_id: int, student: StudentCreate, db: Session = Depends(get_db)):
    """Update a student's information."""
    # Placeholder implementation
    return {
        "id": student_id,
        "name": student.name,
        "roll_number": student.roll_number,
        "department": student.department,
        "semester": student.semester,
        "enrolled_at": datetime.utcnow(),
        "is_active": True
    }


@router.delete("/{student_id}")
def delete_student(student_id: int, db: Session = Depends(get_db)):
    """Soft delete a student."""
    # Placeholder implementation
    return {"message": "Student soft deleted successfully"}
