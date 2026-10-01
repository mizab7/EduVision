"""
Yawn detection module using Mouth Aspect Ratio (MAR) with MediaPipe 1.0 Tasks API.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque
import time

class YawnDetector:
    """
    Detects yawns based on mouth aspect ratio.
    """
    def __init__(self, mar_threshold: float = 0.6):
        self.mar_threshold = mar_threshold
        self.yawn_history = deque(maxlen=50)
        self.is_currently_yawning = False

    def calculate_mar(self, mouth_landmarks: List[Tuple[float, float]]) -> float:
        """Calculates Mouth Aspect Ratio."""
        if len(mouth_landmarks) < 4:
            return 0.0
            
        # For 4 points: 0 is left, 1 is right, 2 is upper, 3 is lower
        v1 = np.linalg.norm(np.array(mouth_landmarks[2]) - np.array(mouth_landmarks[3]))
        h = np.linalg.norm(np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[1]))
        
        if h == 0:
            return 0.0
        return v1 / h

    def detect(self, face_landmarks) -> Dict[str, Any]:
        """Detects yawns from face_landmarks (list of NormalizedLandmark)."""
        if not face_landmarks or len(face_landmarks) < 309:
            return {
                'mar': 0.0,
                'is_yawning': False,
                'yawn_count': 0,
                'yawn_frequency': 0
            }
            
        mouth_indices = [78, 308, 13, 14] # left, right, upper, lower
        mouth_landmarks = [(face_landmarks[i].x, face_landmarks[i].y) for i in mouth_indices]
        mar = self.calculate_mar(mouth_landmarks)
        
        is_yawning = mar > self.mar_threshold
        
        if is_yawning and not self.is_currently_yawning:
            self.yawn_history.append(time.time())
            
        self.is_currently_yawning = is_yawning
        
        current_time = time.time()
        recent_yawns = [t for t in self.yawn_history if current_time - t <= 300.0] # 5 mins
        yawn_count = len(recent_yawns)
        
        # Yawns per hour
        yawn_frequency = yawn_count * 12 
        
        return {
            'mar': mar,
            'is_yawning': is_yawning,
            'yawn_count': yawn_count,
            'yawn_frequency': yawn_frequency
        }
