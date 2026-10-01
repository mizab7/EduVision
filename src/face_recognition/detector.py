"""
Face detection module for EduVision AI.
Uses OpenCV YuNet with multi-threading protection and normalized inference resolution.
"""

import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import urllib.request
import cv2
import numpy as np


class FaceDetector:
    """
    Detects faces in frames using OpenCV's DNN YuNet model.
    Thread-safe and resolution-invariant.
    """
    YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    def __init__(self, model_name: str = "yunet", confidence_threshold: float = 0.5):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.lock = threading.Lock()
        
        project_root = Path(__file__).resolve().parent.parent.parent
        self.models_dir = project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        
        self._ensure_model_exists()
        
        # Initialize detector with default resolution
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (640, 480),
            score_threshold=self.confidence_threshold,
            nms_threshold=0.3,
            top_k=5000
        )
        self.current_input_size = (640, 480)

    def _ensure_model_exists(self):
        """Downloads YuNet weights if not present."""
        if not self.model_path.exists():
            print(f"Downloading YuNet face detection weights to {self.model_path}...")
            urllib.request.urlretrieve(self.YUNET_URL, str(self.model_path))

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in the given BGR frame safely across multiple threads.
        Scales large frames to ~640px width for fast 60+ FPS inference,
        then rescales coordinates back to the original resolution.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        
        # Normalize inference size for extreme stability and speed
        target_w = 640 if w > 640 else w
        scale = target_w / float(w)
        target_h = int(h * scale)

        with self.lock:
            try:
                if scale < 1.0:
                    small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                else:
                    small = frame

                if (target_w, target_h) != self.current_input_size:
                    self.detector.setInputSize((target_w, target_h))
                    self.current_input_size = (target_w, target_h)

                _, faces = self.detector.detect(small)
            except Exception as e:
                # Catch any transient backend buffer assertion and return empty gracefully
                return []

        results = []
        if faces is not None:
            inv_scale = 1.0 / scale
            for face in faces:
                # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
                score = float(face[-1])
                
                # Rescale to original frame dimensions
                x = face[0] * inv_scale
                y = face[1] * inv_scale
                bw = face[2] * inv_scale
                bh = face[3] * inv_scale
                
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(w, int(x + bw))
                y2 = min(h, int(y + bh))
                
                landmarks = {
                    "right_eye": (int(face[4] * inv_scale), int(face[5] * inv_scale)),
                    "left_eye": (int(face[6] * inv_scale), int(face[7] * inv_scale)),
                    "nose_tip": (int(face[8] * inv_scale), int(face[9] * inv_scale)),
                    "right_mouth": (int(face[10] * inv_scale), int(face[11] * inv_scale)),
                    "left_mouth": (int(face[12] * inv_scale), int(face[13] * inv_scale)),
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
