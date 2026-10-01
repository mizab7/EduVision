"""
Alert Rule Engine for EduVision AI.
Monitors real-time classroom engagement and computer vision metrics,
triggers threshold-based pedagogical alerts, manages cooldowns,
and records alerts to the database.
"""

import time
import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session as DBSession

from src.database.models import AlertLog, Session as ClassSession
from src.database.connection import SessionLocal
from config.settings import get_settings


class AlertSeverity:
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class AlertType:
    CLASS_ENGAGEMENT_DROP = "CLASS_ENGAGEMENT_DROP"
    PHONE_DISTRACTION = "PHONE_DISTRACTION"
    DROWSINESS_SLEEP = "DROWSINESS_SLEEP"
    YAWNING_FATIGUE = "YAWNING_FATIGUE"
    PROLONGED_LOOKING_AWAY = "PROLONGED_LOOKING_AWAY"


class AlertRuleEngine:
    """
    Evaluates real-time student and class metrics against pedagogical alert rules.
    Maintains a cooldown window to prevent notification spam.
    """

    def __init__(self, cooldown_seconds: float = 60.0):
        self.cooldown_seconds = cooldown_seconds
        self.settings = get_settings()
        # Track last alert timestamp per rule key (e.g. "PHONE_DISTRACTION:student_3")
        self._last_alert_time: Dict[str, float] = {}

    def _can_fire(self, alert_key: str) -> bool:
        now = time.time()
        last = self._last_alert_time.get(alert_key, 0.0)
        if now - last >= self.cooldown_seconds:
            self._last_alert_time[alert_key] = now
            return True
        return False

    def evaluate_classroom(
        self,
        class_engagement_index: float,
        students_analysis: List[Dict[str, Any]],
        session_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates current class engagement and student metrics.
        Returns list of newly triggered alerts and persists them to the database.
        """
        new_alerts = []

        # Rule 1: Class Engagement Index (CEI) Drop
        drop_threshold = float(getattr(self.settings, "ENGAGEMENT_DROP_THRESHOLD", 45.0))
        if class_engagement_index < drop_threshold and len(students_analysis) > 0:
            key = f"{AlertType.CLASS_ENGAGEMENT_DROP}:class"
            if self._can_fire(key):
                severity = AlertSeverity.CRITICAL if class_engagement_index < 35.0 else AlertSeverity.WARNING
                msg = (
                    f"Class Engagement Index dropped to {class_engagement_index:.1f}% "
                    f"(below {drop_threshold}% threshold). Immediate pedagogical intervention recommended."
                )
                rec = (
                    "Pause lecture and initiate a quick Think-Pair-Share or interactive polling question "
                    "to reactivate student attention."
                )
                alert_dict = {
                    "alert_type": AlertType.CLASS_ENGAGEMENT_DROP,
                    "severity": severity,
                    "message": msg,
                    "recommendation": rec,
                    "session_id": session_id,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
                new_alerts.append(alert_dict)
                self._persist_alert(alert_dict)

        # Student-level rules
        drowsy_count = 0
        phone_count = 0

        for student in students_analysis:
            s_id = student.get("student_id") or student.get("name", "Unknown")
            name = student.get("name", "Student")
            score = student.get("score", 100.0)
            gaze = student.get("gaze", "CENTER")
            posture = student.get("posture", "FORWARD")
            is_drowsy = student.get("is_drowsy", False)
            is_yawning = student.get("is_yawning", False)
            phone = student.get("phone_detected", False)

            # Rule 2: Active Phone Usage
            if phone:
                phone_count += 1
                key = f"{AlertType.PHONE_DISTRACTION}:{s_id}"
                if self._can_fire(key):
                    alert_dict = {
                        "alert_type": AlertType.PHONE_DISTRACTION,
                        "severity": AlertSeverity.WARNING,
                        "message": f"📱 Phone distraction detected near '{name}'. Individual score dropped to {score:.1f}%.",
                        "recommendation": "Gently redirect class attention or remind students to keep devices facedown.",
                        "session_id": session_id,
                        "timestamp": datetime.datetime.utcnow().isoformat()
                    }
                    new_alerts.append(alert_dict)
                    self._persist_alert(alert_dict)

            # Rule 3: Drowsiness / Sleepiness
            if is_drowsy:
                drowsy_count += 1
                key = f"{AlertType.DROWSINESS_SLEEP}:{s_id}"
                if self._can_fire(key):
                    alert_dict = {
                        "alert_type": AlertType.DROWSINESS_SLEEP,
                        "severity": AlertSeverity.WARNING,
                        "message": f"😴 Student '{name}' exhibits prolonged eye closure / signs of micro-sleep.",
                        "recommendation": "Incorporate cold calling with an easy question or invite student participation.",
                        "session_id": session_id,
                        "timestamp": datetime.datetime.utcnow().isoformat()
                    }
                    new_alerts.append(alert_dict)
                    self._persist_alert(alert_dict)

            # Rule 4: Yawning / Fatigue
            if is_yawning:
                key = f"{AlertType.YAWNING_FATIGUE}:{s_id}"
                if self._can_fire(key):
                    alert_dict = {
                        "alert_type": AlertType.YAWNING_FATIGUE,
                        "severity": AlertSeverity.INFO,
                        "message": f"🥱 Yawning detected for '{name}', indicating cognitive exhaustion.",
                        "recommendation": "Increase vocal modulation or shift to visual diagrams/slides.",
                        "session_id": session_id,
                        "timestamp": datetime.datetime.utcnow().isoformat()
                    }
                    new_alerts.append(alert_dict)
                    self._persist_alert(alert_dict)

            # Rule 5: Severe individual disengagement
            if score < 30.0 and posture != "FORWARD":
                key = f"{AlertType.PROLONGED_LOOKING_AWAY}:{s_id}"
                if self._can_fire(key):
                    alert_dict = {
                        "alert_type": AlertType.PROLONGED_LOOKING_AWAY,
                        "severity": AlertSeverity.INFO,
                        "message": f"👀 Student '{name}' is disengaged ({posture}, score: {score:.1f}%).",
                        "recommendation": "Make direct eye contact or walk towards this zone of the classroom.",
                        "session_id": session_id,
                        "timestamp": datetime.datetime.utcnow().isoformat()
                    }
                    new_alerts.append(alert_dict)
                    self._persist_alert(alert_dict)

        return new_alerts

    def _persist_alert(self, alert_dict: Dict[str, Any]):
        """Saves triggered alert to the SQLite database."""
        db: DBSession = SessionLocal()
        try:
            log_entry = AlertLog(
                session_id=alert_dict.get("session_id"),
                alert_type=alert_dict["alert_type"],
                message=alert_dict["message"],
                recommendation=alert_dict["recommendation"],
                acknowledged=False
            )
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)
            alert_dict["id"] = log_entry.id
        except Exception as e:
            db.rollback()
            print(f"Error persisting alert log: {e}")
        finally:
            db.close()

    @staticmethod
    def get_recent_alerts(limit: int = 15, unacknowledged_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieves recent alerts from the database."""
        db: DBSession = SessionLocal()
        try:
            query = db.query(AlertLog).order_by(AlertLog.timestamp.desc())
            if unacknowledged_only:
                query = query.filter(AlertLog.acknowledged == False)
            logs = query.limit(limit).all()

            results = []
            for log in logs:
                results.append({
                    "id": log.id,
                    "session_id": log.session_id,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "alert_type": log.alert_type,
                    "message": log.message,
                    "recommendation": log.recommendation,
                    "acknowledged": log.acknowledged
                })
            return results
        finally:
            db.close()

    @staticmethod
    def acknowledge_alert(alert_id: int) -> bool:
        """Marks an alert as acknowledged by the teacher."""
        db: DBSession = SessionLocal()
        try:
            log = db.query(AlertLog).filter(AlertLog.id == alert_id).first()
            if log:
                log.acknowledged = True
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()
