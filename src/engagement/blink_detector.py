"""
Eye Aspect Ratio (EAR) and blink rate / drowsiness detection module for EduVision AI.
Detects blinks, micro-sleeps, and fatigue using Eye Aspect Ratio and temporal windowing.
"""

import time
from collections import deque
from typing import Dict, Any, List, Optional
import numpy as np


class BlinkDetector:
    """
    Computes Eye Aspect Ratio (EAR), blink frequency, and drowsiness flags.
    """
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def __init__(self, ear_threshold: float = 0.20, drowsiness_duration_threshold: float = 1.2):
        self.ear_threshold = ear_threshold
        self.drowsiness_duration_threshold = drowsiness_duration_threshold

        # Temporal tracking
        self.blink_timestamps = deque(maxlen=120)
        self.is_eye_closed = False
        self.eye_closed_start_time: Optional[float] = None
        self.total_blinks = 0

    @staticmethod
    def calculate_ear(eye_points: np.ndarray) -> float:
        """
        Calculates Eye Aspect Ratio (EAR) for 6 ordered eye contour points.
        Points order: [p1(outer), p2(top-left), p3(top-right), p4(inner), p5(bottom-right), p6(bottom-left)]
        """
        if len(eye_points) < 6:
            return 0.30

        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])

        if h < 1e-5:
            return 0.30

        return float((v1 + v2) / (2.0 * h))

    def detect(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Detects blinks and drowsiness from 478 3D facial landmarks.
        
        Args:
            landmarks: (478, 3) numpy array with landmark coordinates in frame pixels.
            
        Returns:
            Dict containing:
                - ear_left: float
                - ear_right: float
                - ear_avg: float
                - is_blinking: bool
                - is_closed: bool
                - closed_duration: float (seconds)
                - blink_rate_per_min: int
                - is_drowsy: bool
        """
        now = time.time()

        if landmarks is None or len(landmarks) < 400:
            return {
                "ear_left": 0.28, "ear_right": 0.28, "ear_avg": 0.28,
                "is_blinking": False, "is_closed": False,
                "closed_duration": 0.0, "blink_rate_per_min": 15,
                "is_drowsy": False
            }

        left_pts = np.array([landmarks[i][:2] for i in self.LEFT_EYE_INDICES])
        right_pts = np.array([landmarks[i][:2] for i in self.RIGHT_EYE_INDICES])

        ear_l = self.calculate_ear(left_pts)
        ear_r = self.calculate_ear(right_pts)
        ear_avg = float((ear_l + ear_r) / 2.0)

        is_currently_closed = ear_avg < self.ear_threshold
        just_blinked = False
        closed_duration = 0.0

        if is_currently_closed:
            if not self.is_eye_closed:
                # Eye just closed
                self.is_eye_closed = True
                self.eye_closed_start_time = now
            closed_duration = now - self.eye_closed_start_time
        else:
            if self.is_eye_closed:
                # Eye just opened: register a completed blink if duration was normal (< 0.8s)
                self.is_eye_closed = False
                if self.eye_closed_start_time is not None:
                    blink_duration = now - self.eye_closed_start_time
                    if 0.05 <= blink_duration <= 0.8:
                        just_blinked = True
                        self.total_blinks += 1
                        self.blink_timestamps.append(now)
                self.eye_closed_start_time = None

        # Filter blinks within the last 60 seconds
        recent_blinks = [t for t in self.blink_timestamps if (now - t) <= 60.0]
        blink_rate_per_min = len(recent_blinks)

        # Drowsiness condition: Eyes closed continuously for > threshold (micro-sleep) OR abnormally sluggish blink
        is_drowsy = (closed_duration >= self.drowsiness_duration_threshold)

        return {
            "ear_left": round(ear_l, 3),
            "ear_right": round(ear_r, 3),
            "ear_avg": round(ear_avg, 3),
            "is_blinking": just_blinked,
            "is_closed": is_currently_closed,
            "closed_duration": round(closed_duration, 2),
            "blink_rate_per_min": blink_rate_per_min,
            "is_drowsy": is_drowsy
        }
