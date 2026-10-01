"""
Face recognition module for EduVision AI.
Uses OpenCV SFace (FaceRecognizerSF) with YuNet face alignment.
Supports 128-d embeddings, cosine similarity matching, and batch identification.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import urllib.request
import cv2
import numpy as np
from src.security.encryption import EmbeddingEncryption


class FaceRecognizer:
    """
    Extracts 128-d face embeddings and performs 1:N face identification.
    """
    SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    # Default SFace threshold for cosine similarity
    DEFAULT_COSINE_THRESHOLD = 0.363

    def __init__(self, threshold: float = 0.363):
        self.threshold = threshold
        project_root = Path(__file__).resolve().parent.parent.parent
        self.models_dir = project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.sface_path = self.models_dir / "face_recognition_sface_2021dec.onnx"
        self.yunet_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        
        self._ensure_models_exist()
        
        # Initialize recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(str(self.sface_path), "")
        # Initialize detector for standalone face extraction
        self.detector = cv2.FaceDetectorYN.create(
            str(self.yunet_path), "", (320, 320), score_threshold=0.5
        )
        self.current_detector_size = (320, 320)
        self.crypto = EmbeddingEncryption()

    def _ensure_models_exist(self):
        """Ensures both SFace and YuNet models are present."""
        if not self.sface_path.exists():
            print(f"Downloading SFace model to {self.sface_path}...")
            urllib.request.urlretrieve(self.SFACE_URL, str(self.sface_path))
        if not self.yunet_path.exists():
            print(f"Downloading YuNet model to {self.yunet_path}...")
            urllib.request.urlretrieve(self.YUNET_URL, str(self.yunet_path))

    def extract_embedding_from_face(self, frame: np.ndarray, face_data: np.ndarray) -> np.ndarray:
        """
        Extracts 128-d embedding from an already detected 15-dim face representation.
        """
        aligned_face = self.recognizer.alignCrop(frame, face_data)
        feature = self.recognizer.feature(aligned_face)
        # Flatten to 1D (128,)
        return feature.flatten().astype(np.float32)

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Detects the most prominent face in an image and extracts its 128-d embedding.
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Invalid face image provided.")

        h, w = face_image.shape[:2]
        if (w, h) != self.current_detector_size:
            self.detector.setInputSize((w, h))
            self.current_detector_size = (w, h)

        _, faces = self.detector.detect(face_image)
        if faces is None or len(faces) == 0:
            raise ValueError("No face detected in the provided image.")

        # Pick the face with largest area or highest score
        best_face = max(faces, key=lambda f: f[2] * f[3])
        return self.extract_embedding_from_face(face_image, best_face)

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Computes cosine similarity between two 128-d embeddings using SFace match.
        """
        feat1 = np.ascontiguousarray(emb1.reshape(1, -1), dtype=np.float32)
        feat2 = np.ascontiguousarray(emb2.reshape(1, -1), dtype=np.float32)
        score = self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)

    def identify(self, face_embedding: np.ndarray, known_embeddings: Dict[Any, np.ndarray]) -> Tuple[Optional[Any], float]:
        """
        Identifies a face embedding against a dictionary of {student_id: embedding}.
        
        Returns:
            Tuple of (matched_student_id or None, best_cosine_score)
        """
        if not known_embeddings:
            return None, 0.0

        best_id = None
        best_score = -1.0

        for student_id, enrolled_emb in known_embeddings.items():
            score = self.compare_embeddings(face_embedding, enrolled_emb)
            if score > best_score:
                best_score = score
                best_id = student_id

        if best_score >= self.threshold:
            return best_id, best_score
        return None, best_score
