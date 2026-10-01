"""
Mouth Aspect Ratio (MAR) and yawn detection module for EduVision AI.
Detects yawning episodes and tracks fatigue over time using mouth geometric ratios.
"""

import time
from collections import deque
from typing import Dict, Any, List, Optional
import numpy as np


class YawnDetector:
    """
    Computes Mouth Aspect Ratio (MAR) and flags yawning fatigue episodes.
    """
    # Key mouth landmark indices
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_CORNER = 78
    RIGHT_CORNER = 308

    def __init__(self, mar_threshold: float = 0.55, yawn_duration_threshold: float = 1.0):
        self.mar_threshold = mar_threshold
        self.yawn_duration_threshold = yawn_duration_threshold

        self.yawn_timestamps = deque(maxlen=60)
        self.is_mouth_open = False
        self.mouth_open_start_time: Optional[float] = None
        self.total_yawns = 0

    @staticmethod
    def calculate_mar(upper: np.ndarray, lower: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """
        Calculates Mouth Aspect Ratio: vertical height / horizontal width.
        """
        vertical = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left - right)

        if horizontal < 1e-5:
            return 0.0

        return float(vertical / horizontal)

    def detect(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Detects yawning from 478 3D facial landmarks.
        
        Args:
            landmarks: (478, 3) numpy array with landmark coordinates in frame pixels.
            
        Returns:
            Dict containing:
                - mar: float (mouth aspect ratio)
                - is_yawning: bool (currently in an active yawn)
                - yawn_count: int (yawns in the last 15 minutes)
                - open_duration: float (seconds mouth has been wide open)
        """
        now = time.time()

        if landmarks is None or len(landmarks) < 310:
            return {
                "mar": 0.15,
                "is_yawning": False,
                "yawn_count": 0,
                "open_duration": 0.0
            }

        upper = landmarks[self.UPPER_LIP][:2]
        lower = landmarks[self.LOWER_LIP][:2]
        left = landmarks[self.LEFT_CORNER][:2]
        right = landmarks[self.RIGHT_CORNER][:2]

        mar = self.calculate_mar(upper, lower, left, right)
        is_wide_open = mar >= self.mar_threshold
        open_duration = 0.0
        is_yawn_active = False

        if is_wide_open:
            if not self.is_mouth_open:
                self.is_mouth_open = True
                self.mouth_open_start_time = now
            open_duration = now - self.mouth_open_start_time

            # If mouth is wide open for at least the threshold, classify as yawning
            if open_duration >= self.yawn_duration_threshold:
                is_yawn_active = True
        else:
            if self.is_mouth_open:
                # Mouth just closed: check if the open episode qualified as a completed yawn
                if self.mouth_open_start_time is not None:
                    duration = now - self.mouth_open_start_time
                    if duration >= self.yawn_duration_threshold:
                        self.total_yawns += 1
                        self.yawn_timestamps.append(now)
                self.is_mouth_open = False
                self.mouth_open_start_time = None

        # Filter yawns within the last 15 minutes
        recent_yawns = [t for t in self.yawn_timestamps if (now - t) <= 900.0]

        return {
            "mar": round(mar, 3),
            "is_yawning": is_yawn_active,
            "yawn_count": len(recent_yawns),
            "open_duration": round(open_duration, 2)
        }
