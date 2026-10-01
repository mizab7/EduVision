from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.database.connection import get_db

router = APIRouter()


@router.get("/live")
def get_live_engagement(session_id: int, db: Session = Depends(get_db)):
    """Get live engagement data for an active session."""
    # Placeholder implementation
    return {
        "session_id": session_id,
        "overall_engagement": 85.2,
        "attentiveness": 88.0,
        "drowsy_students": 2
    }


@router.get("/history")
def get_historical_engagement(session_id: int, db: Session = Depends(get_db)):
    """Get historical engagement logs for a past session."""
    # Placeholder implementation
    return [
        {
            "timestamp": "2023-10-27T10:05:00",
            "attentiveness_score": 85.0,
            "phone_detected": False
        },
        {
            "timestamp": "2023-10-27T10:10:00",
            "attentiveness_score": 75.0,
            "phone_detected": True
        }
    ]
