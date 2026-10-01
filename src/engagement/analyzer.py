"""
Comprehensive real-time student engagement analyzer for EduVision AI.
Coordinates face mesh extraction, gaze tracking, head pose estimation,
blink/drowsiness detection, yawn detection, phone detection, and attentiveness scoring.
"""

from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from src.engagement.mesh import FaceMeshExtractor
from src.engagement.gaze_tracker import GazeTracker
from src.engagement.head_pose import HeadPoseEstimator
from src.engagement.blink_detector import BlinkDetector
from src.engagement.yawn_detector import YawnDetector
from src.engagement.phone_detector import PhoneDetector
from src.engagement.scorer import AttentivenessScorer


class EngagementAnalyzer:
    """
    Central analyzer orchestrating all 6 engagement signals:
    Gaze, Head Pose, Blink EAR, Yawn MAR, Phone YOLOv8, and Fusion Attentiveness Score.
    """
    def __init__(self):
        self.mesh_extractor = FaceMeshExtractor.get_instance()
        self.gaze_tracker = GazeTracker()
        self.head_pose_estimator = HeadPoseEstimator()
        self.blink_detector = BlinkDetector()
        self.yawn_detector = YawnDetector()
        self.phone_detector = PhoneDetector()
        self.scorer = AttentivenessScorer()

    def analyze_frame(
        self,
        frame: np.ndarray,
        face_detections: List[Dict[str, Any]],
        run_phone_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Processes a video frame and produces full multi-signal engagement metrics for each face.
        
        Args:
            frame: Full BGR video frame
            face_detections: List of dicts with 'bbox': [x1, y1, x2, y2], 'confidence': float
            run_phone_detection: Whether to run YOLO phone inference on this frame
            
        Returns:
            Dict containing:
                - students: List of per-student engagement dictionaries
                - phones: List of detected phone bounding boxes
                - class_engagement_index: float (0 - 100)
                - active_alerts: List of classroom-level warning messages
        """
        if frame is None or frame.size == 0 or not face_detections:
            # Check for phones even if no faces
            phones = self.phone_detector.detect(frame) if run_phone_detection else []
            return {
                "students": [],
                "phones": phones,
                "class_engagement_index": 0.0,
                "active_alerts": ["PHONE_DETECTED"] if phones else []
            }

        face_boxes = [d["bbox"] for d in face_detections]

        # 1. Phone detection (YOLOv8)
        phones = self.phone_detector.detect(frame, face_boxes=face_boxes) if run_phone_detection else []

        students = []
        student_scores = []
        all_alerts = set()

        for idx, det in enumerate(face_detections):
            bbox = det["bbox"]

            # 2. Extract 478 3D landmarks
            mesh_res = self.mesh_extractor.extract_landmarks(frame, bbox)
            if mesh_res is None:
                continue

            lmks = mesh_res["landmarks"]

            # 3. Behavioral signals
            gaze_res = self.gaze_tracker.estimate_gaze(lmks, frame.shape)
            pose_res = self.head_pose_estimator.estimate_pose(lmks, frame.shape)
            blink_res = self.blink_detector.detect(lmks)
            yawn_res = self.yawn_detector.detect(lmks)

            phone_present = any(p.get("near_face_idx") == idx for p in phones)

            # 4. Multi-signal fusion
            score_res = self.scorer.compute_student_score(
                gaze_data=gaze_res,
                pose_data=pose_res,
                blink_data=blink_res,
                yawn_data=yawn_res,
                phone_detected=phone_present
            )

            student_scores.append(score_res["overall_score"])
            for alert in score_res["alerts"]:
                all_alerts.add(alert)

            students.append({
                "face_idx": idx,
                "bbox": bbox,
                "score": score_res["overall_score"],
                "level": score_res["engagement_level"],
                "breakdown": score_res["breakdown"],
                "gaze": gaze_res,
                "pose": pose_res,
                "blink": blink_res,
                "yawn": yawn_res,
                "phone_detected": phone_present,
                "alerts": score_res["alerts"],
                "landmarks": lmks
            })

        class_index = self.scorer.compute_class_engagement(student_scores)

        return {
            "students": students,
            "phones": phones,
            "class_engagement_index": class_index,
            "active_alerts": list(all_alerts)
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        analysis: Dict[str, Any],
        show_mesh: bool = False
    ) -> np.ndarray:
        """
        Draws clean visual engagement overlays on the frame:
        - Bounding boxes colored by engagement level (Green=High, Yellow=Medium, Red=Low/Critical)
        - Attention score badges
        - Gaze & Iris tracking points
        - Phone detection boxes in bright magenta
        - Top classroom HUD banner
        """
        out = frame.copy()

        # 1. Draw phone detections
        for phone in analysis.get("phones", []):
            px1, py1, px2, py2 = phone["bbox"]
            cv2.rectangle(out, (px1, py1), (px2, py2), (255, 0, 255), 2)
            cv2.putText(
                out, f"PHONE ({phone['confidence']:.2f})",
                (px1, max(15, py1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
            )

        # 2. Draw per-student engagement indicators
        for student in analysis.get("students", []):
            bbox = student["bbox"]
            score = student["score"]
            level = student["level"]
            alerts = student["alerts"]

            # Color coding: Green (High), Yellow (Medium), Red (Low/Critical)
            if score >= 75:
                color = (0, 230, 115)       # Green
            elif score >= 55:
                color = (0, 215, 255)       # Amber / Yellow
            elif score >= 35:
                color = (0, 140, 255)       # Orange
            else:
                color = (0, 0, 255)         # Red

            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Iris dots
            iris = student["gaze"].get("iris_positions", {})
            if "left" in iris and "right" in iris:
                cv2.circle(out, iris["left"], 3, (0, 255, 255), -1)
                cv2.circle(out, iris["right"], 3, (0, 255, 255), -1)

            # Badge with score and posture
            posture = student["pose"].get("posture_label", "FORWARD")
            badge_text = f"ATTN: {int(score)}% [{level}] | {posture}"
            if alerts:
                badge_text += f" | {alerts[0]}"

            tag_w = len(badge_text) * 8 + 12
            cv2.rectangle(out, (x1, max(0, y1 - 22)), (x1 + tag_w, y1), color, -1)
            cv2.putText(
                out, badge_text, (x1 + 6, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA
            )

        # 3. Classroom Engagement HUD (top-left)
        cei = analysis.get("class_engagement_index", 0.0)
        hud_bg = (15, 23, 42) # Slate 900
        cv2.rectangle(out, (12, 12), (320, 68), hud_bg, -1)
        cv2.rectangle(out, (12, 12), (320, 68), (51, 65, 85), 1)

        cei_color = (0, 230, 115) if cei >= 75 else ((0, 215, 255) if cei >= 55 else (0, 0, 255))
        cv2.putText(
            out, f"Class Engagement Index: {cei:.1f}%",
            (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cei_color, 2, cv2.LINE_AA
        )

        active_alerts = analysis.get("active_alerts", [])
        if active_alerts:
            alert_str = " | ".join(active_alerts[:2])
            cv2.putText(
                out, f"Warning: {alert_str}",
                (22, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 100, 255), 1, cv2.LINE_AA
            )
        else:
            cv2.putText(
                out, "Status: All students attentive",
                (22, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA
            )

        return out
