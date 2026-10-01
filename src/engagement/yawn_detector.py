"""
Yawn detection module using Mouth Aspect Ratio (MAR).
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

    def calculate_mar(self, mouth_landmarks: List[Tuple[int, int]]) -> float:
        """Calculates Mouth Aspect Ratio."""
        if len(mouth_landmarks) < 6:
            return 0.0
            
        v1 = np.linalg.norm(np.array(mouth_landmarks[1]) - np.array(mouth_landmarks[7]))
        v2 = np.linalg.norm(np.array(mouth_landmarks[2]) - np.array(mouth_landmarks[6]))
        v3 = np.linalg.norm(np.array(mouth_landmarks[3]) - np.array(mouth_landmarks[5]))
        h = np.linalg.norm(np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[4]))
        
        if h == 0:
            return 0.0
        return (v1 + v2 + v3) / (3.0 * h)

    def detect(self, face_landmarks: Dict[str, List[Tuple[int, int]]]) -> Dict[str, Any]:
        """Detects yawns."""
        mouth_landmarks = face_landmarks.get('mouth', [])
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
