from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from src.database.connection import get_db
from src.database.models import Session as ClassSession

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
    ended_at: Optional[datetime] = None
    class_engagement_index: Optional[float] = None

    class Config:
        from_attributes = True


@router.post("/", response_model=SessionResponse)
def start_session(session_data: SessionCreate, db: DBSession = Depends(get_db)):
    """Start a new class session and persist to database."""
    session_obj = ClassSession(
        subject=session_data.subject,
        faculty_name=session_data.faculty_name,
        classroom=session_data.classroom,
        started_at=datetime.utcnow()
    )
    db.add(session_obj)
    db.commit()
    db.refresh(session_obj)
    return session_obj


@router.put("/{session_id}/end", response_model=SessionResponse)
def end_session(session_id: int, db: DBSession = Depends(get_db)):
    """End an active class session."""
    session_obj = db.query(ClassSession).filter(ClassSession.id == session_id).first()
    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session_obj.ended_at = datetime.utcnow()
    db.commit()
    db.refresh(session_obj)
    return session_obj


@router.get("/", response_model=List[SessionResponse])
def list_sessions(db: DBSession = Depends(get_db)):
    """List all classroom sessions."""
    return db.query(ClassSession).order_by(ClassSession.started_at.desc()).all()


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: int, db: DBSession = Depends(get_db)):
    """Retrieve details of a specific session."""
    session_obj = db.query(ClassSession).filter(ClassSession.id == session_id).first()
    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session_obj
