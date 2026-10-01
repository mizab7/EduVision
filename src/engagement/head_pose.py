"""
Head pose estimation module for EduVision AI.
Estimates 3D head orientation (Pitch, Yaw, Roll) using OpenCV solvePnP on key facial landmarks.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


class HeadPoseEstimator:
    """
    Estimates 3D head pose and posture attentiveness using Perspective-n-Point.
    """
    def __init__(self):
        # 3D generic facial model points (in mm, aligned with image coordinate frame)
        # X: Right (+), Y: Down (+), Z: Away (+ into scene)
        self.model_points = np.array([
            [0.0, 0.0, 0.0],          # Nose tip (index 1)
            [0.0, 65.0, -20.0],       # Chin (index 152)
            [-35.0, -35.0, -25.0],    # Viewer left eye (index 33)
            [35.0, -35.0, -25.0],     # Viewer right eye (index 263)
            [-25.0, 35.0, -25.0],     # Viewer left mouth (index 61)
            [25.0, 35.0, -25.0]       # Viewer right mouth (index 291)
        ], dtype=np.float64)

    def estimate_pose(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Estimates Pitch, Yaw, Roll angles and posture attentiveness.
        
        Args:
            landmarks: (478, 3) numpy array with landmark coordinates in frame pixels.
            frame_shape: (H, W) or (H, W, C)
            
        Returns:
            Dict containing:
                - pitch: float (degrees, negative = looking down, positive = looking up)
                - yaw: float (degrees, negative = looking right, positive = looking left)
                - roll: float (degrees, tilt)
                - pose_score: float (0 - 100)
                - is_attentive: bool
                - posture_label: str ('FORWARD', 'LOOKING_DOWN', 'LOOKING_LEFT', 'LOOKING_RIGHT', 'LOOKING_UP')
        """
        if landmarks is None or len(landmarks) < 300:
            return {
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                "pose_score": 85.0, "is_attentive": True,
                "posture_label": "FORWARD"
            }

        h, w = frame_shape[:2]

        indices = [1, 152, 33, 263, 61, 291]
        image_points = np.array([landmarks[i][:2] for i in indices], dtype=np.float64)

        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            return {
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                "pose_score": 75.0, "is_attentive": True,
                "posture_label": "FORWARD"
            }

        rmat, _ = cv2.Rodrigues(rvec)

        # Robust Euler decomposition
        pitch = float(np.degrees(np.arcsin(-np.clip(rmat[1, 2], -1.0, 1.0))))
        yaw = float(np.degrees(np.arctan2(rmat[0, 2], rmat[2, 2])))
        roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[1, 1])))

        # Posture classification
        if pitch < -25.0:
            posture_label = "LOOKING_DOWN"
        elif pitch > 25.0:
            posture_label = "LOOKING_UP"
        elif yaw > 30.0:
            posture_label = "LOOKING_LEFT"
        elif yaw < -30.0:
            posture_label = "LOOKING_RIGHT"
        else:
            posture_label = "FORWARD"

        is_attentive = (abs(pitch) <= 22.0) and (abs(yaw) <= 25.0)

        # Attentiveness score penalty based on angular deviation
        deviation = max(0.0, abs(pitch) - 10.0) * 1.5 + max(0.0, abs(yaw) - 12.0) * 1.5
        pose_score = max(20.0, min(100.0, 100.0 - deviation))

        return {
            "pitch": round(pitch, 1),
            "yaw": round(yaw, 1),
            "roll": round(roll, 1),
            "pose_score": round(pose_score, 1),
            "is_attentive": is_attentive,
            "posture_label": posture_label
        }
