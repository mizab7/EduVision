"""
Phone usage detection module for EduVision AI.
Uses YOLOv8 nano object detection targeting cell phone class (COCO 67)
and correlates phone bounding boxes with student locations.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


class PhoneDetector:
    """
    Detects cell phones and associates detections with corresponding student positions.
    """
    CELL_PHONE_CLASS_ID = 67

    def __init__(self, confidence_threshold: float = 0.40):
        self.confidence_threshold = confidence_threshold
        model_path = Path(__file__).resolve().parent.parent.parent / "models" / "yolov8n.pt"
        self.model = YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")

    def detect(self, frame: np.ndarray, face_boxes: Optional[List[list]] = None) -> List[Dict[str, Any]]:
        """
        Detects cell phones in the current video frame.
        
        Args:
            frame: Full BGR frame
            face_boxes: Optional list of student face bboxes [[x1, y1, x2, y2], ...]
            
        Returns:
            List of detected phone objects:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - near_face_idx: Optional[int] (index of the nearest face)
        """
        if frame is None or frame.size == 0:
            return []

        # Run YOLO inference specifically targeting cell phone class
        results = self.model(
            frame,
            classes=[self.CELL_PHONE_CLASS_ID],
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                px1, py1, px2, py2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

                # Associate with nearest face if face boxes provided
                near_face_idx = None
                if face_boxes:
                    phone_center = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
                    min_dist = float("inf")
                    for f_idx, fb in enumerate(face_boxes):
                        fx1, fy1, fx2, fy2 = fb
                        face_center = np.array([(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0])
                        dist = np.linalg.norm(phone_center - face_center)
                        # Check horizontal proximity (phone is typically below the head)
                        horiz_overlap = not (px2 < fx1 - 100 or px1 > fx2 + 100)
                        if horiz_overlap and dist < min_dist:
                            min_dist = dist
                            near_face_idx = f_idx

                detections.append({
                    "bbox": [px1, py1, px2, py2],
                    "confidence": round(conf, 3),
                    "near_face_idx": near_face_idx
                })

        return detections
