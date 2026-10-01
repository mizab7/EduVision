"""
Blink detection module using Eye Aspect Ratio (EAR) with MediaPipe 1.0 Tasks API.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque
import time

class BlinkDetector:
    """
    Detects blinks and tracks blink rate.
    """
    def __init__(self, ear_threshold: float = 0.2):
        self.ear_threshold = ear_threshold
        self.blink_history = deque(maxlen=100) # stores timestamps of blinks
        self.was_closed = False

    def calculate_ear(self, eye_landmarks: List[Tuple[float, float]]) -> float:
        """Calculates Eye Aspect Ratio."""
        if len(eye_landmarks) < 6:
            return 0.0
            
        # Euclidean distances
        v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def detect(self, face_landmarks) -> Dict[str, Any]:
        """Detects blinking from left and right eye landmarks. face_landmarks is a list of NormalizedLandmark."""
        if not face_landmarks or len(face_landmarks) < 468:
            return {
                'ear_left': 0.0,
                'ear_right': 0.0,
                'ear_avg': 0.0,
                'is_blinking': False,
                'blink_rate_per_min': 0,
                'is_drowsy': False
            }
            
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        left_eye = [(face_landmarks[i].x, face_landmarks[i].y) for i in left_eye_indices]
        right_eye = [(face_landmarks[i].x, face_landmarks[i].y) for i in right_eye_indices]
        
        ear_left = self.calculate_ear(left_eye)
        ear_right = self.calculate_ear(right_eye)
        ear_avg = (ear_left + ear_right) / 2.0
        
        is_closed = ear_avg > 0 and ear_avg < self.ear_threshold
        is_blinking = False
        
        if is_closed and not self.was_closed:
            is_blinking = True
            self.blink_history.append(time.time())
            
        self.was_closed = is_closed
        
        # Calculate rate per min
        current_time = time.time()
        # filter blinks in last 60 seconds
        recent_blinks = [t for t in self.blink_history if current_time - t <= 60.0]
        blink_rate_per_min = len(recent_blinks)
        
        # Heuristic for drowsiness
        is_drowsy = blink_rate_per_min > 25 or blink_rate_per_min < 5
        
        return {
            'ear_left': ear_left,
            'ear_right': ear_right,
            'ear_avg': ear_avg,
            'is_blinking': is_blinking,
            'blink_rate_per_min': blink_rate_per_min,
            'is_drowsy': is_drowsy
        }
