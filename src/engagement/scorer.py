"""
Attentiveness scoring module.
"""

from typing import Dict, List, Optional

class AttentivenessScorer:
    """
    Calculates engagement and attentiveness scores.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        if weights is None:
            self.weights = {
                'gaze': 0.3,
                'head_pose': 0.25,
                'blink': 0.15,
                'yawn': 0.15,
                'phone': 0.15
            }
        else:
            self.weights = weights
            
        # Normalize weights if they don't sum to 1
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

    def compute_student_score(self, gaze_data: Dict, pose_data: Dict, blink_data: Dict, yawn_data: Dict, phone_detected: bool) -> float:
        """Computes a single student's attentiveness score (0-100)."""
        score = 100.0
        
        # Gaze contribution
        gaze_score = gaze_data.get('gaze_score', 50)
        score -= (100 - gaze_score) * self.weights.get('gaze', 0.3)
        
        # Pose contribution
        pose_score = pose_data.get('pose_score', 50)
        score -= (100 - pose_score) * self.weights.get('head_pose', 0.25)
        
        # Blink penalty (if drowsy)
        if blink_data.get('is_drowsy', False):
            score -= 100 * self.weights.get('blink', 0.15)
            
        # Yawn penalty
        if yawn_data.get('is_yawning', False):
            score -= 100 * self.weights.get('yawn', 0.15)
            
        # Phone penalty
        if phone_detected:
            score -= 100 * self.weights.get('phone', 0.15)
            
        return max(0.0, min(100.0, score))

    def compute_class_engagement(self, student_scores: List[float]) -> float:
        """Computes overall class engagement average."""
        if not student_scores:
            return 0.0
        return sum(student_scores) / len(student_scores)

    def get_engagement_level(self, score: float) -> str:
        """Categorizes score into qualitative levels."""
        if score >= 80:
            return 'High'
        elif score >= 60:
            return 'Medium'
        elif score >= 40:
            return 'Low'
        else:
            return 'Critical'
