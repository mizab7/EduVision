"""
Analytics & Predictive Intelligence API routes for EduVision AI.
Provides institutional overview KPIs, temporal engagement trends,
behavioral telemetry distributions, and machine learning at-risk predictions.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session as DBSession

from src.database.connection import get_db
from src.analytics.engine import AnalyticsEngine

router = APIRouter()
analytics_engine = AnalyticsEngine()


@router.get("/overview")
def get_analytics_overview(db: DBSession = Depends(get_db)):
    """Returns macro institutional KPIs across all enrolled students and sessions."""
    try:
        return analytics_engine.get_overview_kpis(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute overview KPIs: {e}")


@router.get("/trends")
def get_engagement_trends(db: DBSession = Depends(get_db)):
    """Returns session-by-session CEI progression and time-of-day attention distribution."""
    try:
        return analytics_engine.get_engagement_trends(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute engagement trends: {e}")


@router.get("/behavioral")
def get_behavioral_distribution(db: DBSession = Depends(get_db)):
    """Returns aggregated computer vision telemetry breakdowns (gaze, posture, phone, sleep)."""
    try:
        return analytics_engine.get_behavioral_signals_distribution(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute behavioral distribution: {e}")


@router.get("/students-risk")
def get_students_risk_profiles(db: DBSession = Depends(get_db)):
    """
    Evaluates all enrolled students with the scikit-learn machine learning
    disengagement risk model and returns risk scores, factors, and interventions.
    """
    try:
        profiles = analytics_engine.get_all_students_risk_profiles(db)
        return {
            "total_students": len(profiles),
            "profiles": profiles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate student risk profiles: {e}")


@router.post("/retrain")
def retrain_model(db: DBSession = Depends(get_db)):
    """Retrains the machine learning predictive risk pipeline and updates model weights."""
    try:
        analytics_engine.predictor._load_or_train_baseline()
        analytics_engine.predictor.save_model()
        return {
            "status": "success",
            "message": "Predictive analytics machine learning model retrained successfully!",
            "model_path": analytics_engine.predictor.model_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {e}")


@router.post("/seed-sample-data")
def seed_sample_analytics_data(db: DBSession = Depends(get_db)):
    """Seeds realistic historical lecture sessions and engagement logs for dashboard demonstration."""
    try:
        result = analytics_engine.seed_sample_data(db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to seed analytics sample data: {e}")
