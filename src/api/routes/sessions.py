from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.connection import get_db

router = APIRouter()


class SessionCreate(BaseModel):
    subject: str
    faculty_name: str
    classroom: str


class SessionResponse(BaseModel):
    id: int
    subject: str
    faculty_name: str
    classroom: str
    started_at: datetime
    ended_at: Optional[datetime]
    class_engagement_index: Optional[float]

    class Config:
        from_attributes = True


@router.post("/", response_model=SessionResponse)
def start_session(session: SessionCreate, db: Session = Depends(get_db)):
    """Start a new class session."""
    # Placeholder implementation
    return {
        "id": 1,
        "subject": session.subject,
        "faculty_name": session.faculty_name,
        "classroom": session.classroom,
        "started_at": datetime.utcnow(),
        "ended_at": None,
        "class_engagement_index": None
    }


@router.put("/{session_id}/end", response_model=SessionResponse)
def end_session(session_id: int, db: Session = Depends(get_db)):
    """End an active class session."""
    # Placeholder implementation
    return {
        "id": session_id,
        "subject": "Computer Vision",
        "faculty_name": "Dr. Smith",
        "classroom": "Room 101",
        "started_at": datetime.utcnow(), # simplified for placeholder
        "ended_at": datetime.utcnow(),
        "class_engagement_index": 82.5
    }


@router.get("/", response_model=List[SessionResponse])
def list_sessions(db: Session = Depends(get_db)):
    """List all sessions."""
    # Placeholder implementation
    return []


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: int, db: Session = Depends(get_db)):
    """Retrieve details of a specific session."""
    # Placeholder implementation
    return {
        "id": session_id,
        "subject": "Computer Vision",
        "faculty_name": "Dr. Smith",
        "classroom": "Room 101",
        "started_at": datetime.utcnow(),
        "ended_at": None,
        "class_engagement_index": None
    }
