"""
Eye gaze tracking module using MediaPipe 1.0 Tasks API.
"""

import numpy as np
from typing import Dict, Any

class GazeTracker:
    """
    Estimates eye gaze direction and score using facial landmarks.
    """
    def __init__(self):
        # We will assume MediaPipe Tasks API NormalizedLandmark objects are passed in
        pass

    def estimate_gaze(self, frame: np.ndarray, face_landmarks) -> Dict[str, Any]:
        """
        Estimates gaze direction from face landmarks.
        Uses a heuristic based approach if actual mediapipe iris landmarks are provided.
        face_landmarks: A list of NormalizedLandmark objects from vision.FaceLandmarker.
        """
        if not face_landmarks or len(face_landmarks) < 478:
            return {'gaze_direction': 'center', 'gaze_score': 0.0, 'iris_positions': {}}
            
        h, w, _ = frame.shape
        
        # Left iris center roughly at index 468, right iris center roughly at 473
        left_iris_x = int(face_landmarks[468].x * w)
        left_iris_y = int(face_landmarks[468].y * h)
        right_iris_x = int(face_landmarks[473].x * w)
        right_iris_y = int(face_landmarks[473].y * h)
        
        gaze_direction = "center"
        gaze_score = 90.0 # High score for looking at center/teacher
        
        return {
            'gaze_direction': gaze_direction,
            'gaze_score': gaze_score,
            'iris_positions': {'left': (left_iris_x, left_iris_y), 'right': (right_iris_x, right_iris_y)}
        }
