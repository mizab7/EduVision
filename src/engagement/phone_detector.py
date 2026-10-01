"""
Phone usage detection module.
"""

import numpy as np
from typing import List, Dict, Any, Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class PhoneDetector:
    """
    Detects cell phones using YOLOv8.
    """
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        if YOLO is None:
            print("Warning: ultralytics is not installed. Phone detection will be mocked.")
            self.model = None
        else:
            self.model = YOLO('yolov8n.pt')

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detects phones in the frame."""
        results = []
        if self.model is None:
            return results
            
        # Class 67 is cell phone in COCO
        predictions = self.model(frame, classes=[67], conf=self.confidence_threshold, verbose=False)
        
        for r in predictions:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                results.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'near_student_id': None # Requires spatial logic with face bounding boxes
                })
                
        return results
