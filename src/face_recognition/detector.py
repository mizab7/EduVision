"""
Face detection module for EduVision AI.
Uses OpenCV YuNet (high-performance, native ONNX face detector) with multi-face support.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import urllib.request
import cv2
import numpy as np


class FaceDetector:
    """
    Detects faces in frames using OpenCV's DNN YuNet model.
    """
    YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    def __init__(self, model_name: str = "yunet", confidence_threshold: float = 0.5):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        project_root = Path(__file__).resolve().parent.parent.parent
        self.models_dir = project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        
        self._ensure_model_exists()
        
        # Initialize detector with default resolution
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            score_threshold=self.confidence_threshold,
            nms_threshold=0.3,
            top_k=5000
        )
        self.current_input_size = (320, 320)

    def _ensure_model_exists(self):
        """Downloads YuNet weights if not present."""
        if not self.model_path.exists():
            print(f"Downloading YuNet face detection weights to {self.model_path}...")
            urllib.request.urlretrieve(self.YUNET_URL, str(self.model_path))

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in the given BGR frame.
        
        Returns:
            List of dicts: [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'landmarks': {
                        'right_eye': (x, y),
                        'left_eye': (x, y),
                        'nose_tip': (x, y),
                        'right_mouth': (x, y),
                        'left_mouth': (x, y)
                    }
                }
            ]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if (w, h) != self.current_input_size:
            self.detector.setInputSize((w, h))
            self.current_input_size = (w, h)

        _, faces = self.detector.detect(frame)
        results = []

        if faces is not None:
            for face in faces:
                # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
                x, y, bw, bh = face[0:4]
                score = float(face[-1])
                
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(w, int(x + bw))
                y2 = min(h, int(y + bh))
                
                landmarks = {
                    "right_eye": (int(face[4]), int(face[5])),
                    "left_eye": (int(face[6]), int(face[7])),
                    "nose_tip": (int(face[8]), int(face[9])),
                    "right_mouth": (int(face[10]), int(face[11])),
                    "left_mouth": (int(face[12]), int(face[13])),
                }
                
                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": score,
                    "landmarks": landmarks
                })

        return results

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draws bounding boxes and facial landmarks on a copy of the frame."""
        out_frame = frame.copy()
        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            
            # Bounding box
            cv2.rectangle(out_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Landmarks
            for _, pt in det["landmarks"].items():
                cv2.circle(out_frame, pt, 3, (0, 0, 255), -1)
                
            # Score label
            label = f"Face: {conf:.2f}"
            cv2.putText(out_frame, label, (bbox[0], max(15, bbox[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return out_frame
