"""
Head pose estimation module using MediaPipe 1.0 Tasks API landmarks.
"""

import cv2
import numpy as np
from typing import Dict, Any

class HeadPoseEstimator:
    """
    Estimates head pose (pitch, yaw, roll) using solvePnP.
    """
    def __init__(self):
        # 3D model points (generic face)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def estimate_pose(self, frame: np.ndarray, face_landmarks) -> Dict[str, Any]:
        """
        Calculates pitch, yaw, roll and an attentiveness score.
        face_landmarks should be a list of NormalizedLandmark objects.
        """
        if not face_landmarks or len(face_landmarks) < 468:
            return {'pitch': 0, 'yaw': 0, 'roll': 0, 'pose_score': 0, 'is_attentive': False}
            
        size = frame.shape
        h, w = size[0], size[1]
        
        # Extract the 6 key points
        indices = [1, 152, 33, 263, 61, 291]
        image_points = []
        for idx in indices:
            lm = face_landmarks[idx]
            image_points.append((lm.x * w, lm.y * h))
            
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        image_points_np = np.array(image_points, dtype="double")
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points_np, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return {'pitch': 0, 'yaw': 0, 'roll': 0, 'pose_score': 0, 'is_attentive': False}
            
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        
        # Simple attention heuristic based on angles
        is_attentive = abs(pitch) < 20 and abs(yaw) < 30
        pose_score = 100 if is_attentive else max(0, 100 - (abs(pitch) + abs(yaw)))
        
        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'pose_score': pose_score,
            'is_attentive': is_attentive
        }
