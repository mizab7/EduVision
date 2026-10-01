"""
Attentiveness scoring and multi-signal engagement fusion engine for EduVision AI.
Fuses 5 behavioral indicators (gaze, head pose, blink/drowsiness, yawning, phone usage)
into a unified real-time score (0-100) per student and computes the Class Engagement Index (CEI).
"""

from typing import Dict, List, Any, Optional


class AttentivenessScorer:
    """
    Weighted multi-signal attention scoring engine.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "gaze": 0.30,
            "head_pose": 0.25,
            "blink": 0.15,
            "yawn": 0.15,
            "phone": 0.15
        }
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

    def compute_student_score(
        self,
        gaze_data: Dict[str, Any],
        pose_data: Dict[str, Any],
        blink_data: Dict[str, Any],
        yawn_data: Dict[str, Any],
        phone_detected: bool
    ) -> Dict[str, Any]:
        """
        Fuses all 5 behavioral indicators into a standardized 0-100 attentiveness score.
        
        Returns:
            Dict containing:
                - overall_score: float (0 - 100)
                - engagement_level: 'High', 'Medium', 'Low', 'Critical'
                - breakdown: Dict of normalized sub-scores
                - alerts: List of active behavioral warning flags
        """
        gaze_score = float(gaze_data.get("gaze_score", 80.0))
        pose_score = float(pose_data.get("pose_score", 80.0))

        # Blink / Drowsiness subscore
        is_drowsy = blink_data.get("is_drowsy", False)
        blink_rate = blink_data.get("blink_rate_per_min", 15)
        if is_drowsy:
            blink_subscore = 15.0  # Prolonged eye closure
        elif blink_rate > 30:
            blink_subscore = 50.0  # Rapid blinking / fatigue
        else:
            blink_subscore = 100.0

        # Yawn subscore
        is_yawning = yawn_data.get("is_yawning", False)
        if is_yawning:
            yawn_subscore = 25.0
        else:
            yawn_subscore = 100.0

        # Phone subscore
        phone_subscore = 0.0 if phone_detected else 100.0

        # Weighted composite score
        overall = (
            gaze_score * self.weights["gaze"] +
            pose_score * self.weights["head_pose"] +
            blink_subscore * self.weights["blink"] +
            yawn_subscore * self.weights["yawn"] +
            phone_subscore * self.weights["phone"]
        )

        overall = max(0.0, min(100.0, round(overall, 1)))

        # Determine qualitative level
        if overall >= 75.0:
            level = "High"
        elif overall >= 55.0:
            level = "Medium"
        elif overall >= 35.0:
            level = "Low"
        else:
            level = "Critical"

        # Generate action alerts
        alerts = []
        if phone_detected:
            alerts.append("PHONE_DISTRACTION")
        if is_drowsy:
            alerts.append("DROWSINESS_DETECTED")
        if is_yawning:
            alerts.append("YAWN_FATIGUE")
        if pose_data.get("posture_label") == "LOOKING_DOWN" and not phone_detected:
            alerts.append("HEAD_DROOPING")
        elif not pose_data.get("is_attentive", True):
            alerts.append("LOOKING_AWAY")

        return {
            "overall_score": overall,
            "engagement_level": level,
            "breakdown": {
                "gaze": round(gaze_score, 1),
                "head_pose": round(pose_score, 1),
                "blink": round(blink_subscore, 1),
                "yawn": round(yawn_subscore, 1),
                "phone": round(phone_subscore, 1)
            },
            "alerts": alerts
        }

    def compute_class_engagement(self, student_scores: List[float]) -> float:
        """
        Computes the aggregate Class Engagement Index (CEI) from individual student scores.
        """
        if not student_scores:
            return 0.0
        return round(sum(student_scores) / float(len(student_scores)), 1)
