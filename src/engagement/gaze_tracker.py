"""
Eye gaze tracking module for EduVision AI.
Calculates iris displacement relative to eye contours to determine gaze direction and attentiveness.
"""

from typing import Dict, Any, Tuple
import numpy as np


class GazeTracker:
    """
    Estimates eye gaze direction and score using 478 MediaPipe/ONNX FaceMesh landmarks.
    """

    def __init__(self):
        pass

    def estimate_gaze(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Estimates gaze direction and attention score from 478 3D facial landmarks.
        
        Args:
            landmarks: (478, 3) numpy array with landmark coordinates in frame pixels.
            frame_shape: (H, W) or (H, W, C)
            
        Returns:
            Dict containing:
                - gaze_direction: 'CENTER', 'LEFT', 'RIGHT', 'UP', 'DOWN'
                - gaze_score: float (0 - 100)
                - iris_positions: {'left': (x, y), 'right': (x, y)}
                - ratios: {'horizontal': float, 'vertical': float}
        """
        if landmarks is None or len(landmarks) < 478:
            return {
                "gaze_direction": "CENTER",
                "gaze_score": 70.0,
                "iris_positions": {},
                "ratios": {"horizontal": 0.5, "vertical": 0.5}
            }

        # Left eye: 33 (outer corner), 133 (inner corner), 468 (iris center)
        # 159 (upper eyelid), 145 (lower eyelid)
        p33 = landmarks[33][:2]
        p133 = landmarks[133][:2]
        p468 = landmarks[468][:2]
        p159 = landmarks[159][:2]
        p145 = landmarks[145][:2]

        # Right eye: 362 (inner corner), 263 (outer corner), 473 (iris center)
        # 386 (upper eyelid), 374 (lower eyelid)
        p362 = landmarks[362][:2]
        p263 = landmarks[263][:2]
        p473 = landmarks[473][:2]
        p386 = landmarks[386][:2]
        p374 = landmarks[374][:2]

        # Horizontal ratios: relative position of iris across the eye width
        d_l_total = np.linalg.norm(p133 - p33)
        d_l_iris = np.linalg.norm(p468 - p33)
        ratio_l_h = d_l_iris / max(1e-5, d_l_total)

        d_r_total = np.linalg.norm(p263 - p362)
        d_r_iris = np.linalg.norm(p473 - p362)
        ratio_r_h = d_r_iris / max(1e-5, d_r_total)

        avg_ratio_h = float((ratio_l_h + ratio_r_h) / 2.0)

        # Vertical ratios: relative position of iris between upper and lower eyelids
        d_l_v_total = np.linalg.norm(p145 - p159)
        d_l_v_iris = np.linalg.norm(p468 - p159)
        ratio_l_v = d_l_v_iris / max(1e-5, d_l_v_total)

        d_r_v_total = np.linalg.norm(p374 - p386)
        d_r_v_iris = np.linalg.norm(p473 - p386)
        ratio_r_v = d_r_v_iris / max(1e-5, d_r_v_total)

        avg_ratio_v = float((ratio_l_v + ratio_r_v) / 2.0)

        # Classify gaze direction
        # Horizontal thresholds: Normal forward gaze is ~0.42 to 0.58
        if avg_ratio_h < 0.38:
            direction = "RIGHT"
            h_pen = abs(0.5 - avg_ratio_h) * 120
        elif avg_ratio_h > 0.62:
            direction = "LEFT"
            h_pen = abs(0.5 - avg_ratio_h) * 120
        elif avg_ratio_v < 0.25:
            direction = "UP"
            h_pen = abs(0.45 - avg_ratio_v) * 90
        elif avg_ratio_v > 0.65:
            direction = "DOWN"
            h_pen = abs(0.45 - avg_ratio_v) * 120
        else:
            direction = "CENTER"
            h_pen = 0.0

        gaze_score = max(20.0, min(100.0, 100.0 - h_pen))

        return {
            "gaze_direction": direction,
            "gaze_score": round(gaze_score, 1),
            "iris_positions": {
                "left": (int(p468[0]), int(p468[1])),
                "right": (int(p473[0]), int(p473[1]))
            },
            "ratios": {
                "horizontal": round(avg_ratio_h, 3),
                "vertical": round(avg_ratio_v, 3)
            }
        }
