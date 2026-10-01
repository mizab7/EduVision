"""
Face detection module using MediaPipe 1.0 Tasks API.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None
    vision = None


class FaceDetector:
    """
    Detects faces in an image using MediaPipe Tasks API.
    """
    def __init__(self, model_name: str = 'retinaface', confidence_threshold: float = 0.5):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        if self.model_name in ['mediapipe', 'retinaface']:
            if mp is None or vision is None:
                raise ImportError("MediaPipe is required for face detection.")
            options = vision.FaceDetectorOptions(
                min_detection_confidence=self.confidence_threshold,
                num_faces=10
            )
            self.detector = vision.FaceDetector.create_from_options(options)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detects faces and returns bounding boxes and landmarks."""
        results = []
        if self.model_name in ['mediapipe', 'retinaface']:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            out = self.detector.detect(mp_image)
            
            if out.detections:
                h, w, _ = frame.shape
                for detection in out.detections:
                    bbox = detection.bounding_box
                    x1 = bbox.origin_x
                    y1 = bbox.origin_y
                    x2 = x1 + bbox.width
                    y2 = y1 + bbox.height
                    
                    confidence = detection.categories[0].score if detection.categories else 0.0
                    
                    landmarks = {}
                    if hasattr(detection, 'keypoints') and detection.keypoints:
                        for i, kp in enumerate(detection.keypoints):
                            landmarks[f'kp_{i}'] = (int(kp.x * w), int(kp.y * h))
                            
                    results.append({
                        'bbox': [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                        'confidence': float(confidence),
                        'landmarks': landmarks
                    })
        return results

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draws bounding boxes and landmarks on the frame."""
        out_frame = frame.copy()
        for det in detections:
            bbox = det['bbox']
            cv2.rectangle(out_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            for name, pt in det['landmarks'].items():
                cv2.circle(out_frame, pt, 2, (0, 0, 255), -1)
                
            conf = det['confidence']
            cv2.putText(out_frame, f"{conf:.2f}", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out_frame
