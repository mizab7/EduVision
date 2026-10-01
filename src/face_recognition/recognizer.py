"""
Face recognition module.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import math

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


class FaceRecognizer:
    """
    Extracts face embeddings and identifies known faces.
    """
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        if FaceAnalysis is None:
            print("Warning: InsightFace is not installed. Face recognition will mock embeddings.")
            self.app = None
        else:
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extracts a 512-d embedding from a face image."""
        if self.app is None:
            return np.random.rand(512).astype(np.float32)
            
        faces = self.app.get(face_image)
        if not faces:
            raise ValueError("No face detected in the image for embedding extraction.")
        
        # Assume the most prominent face is the first one
        return faces[0].embedding

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes cosine similarity between two embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        if norm_emb1 == 0 or norm_emb2 == 0:
            return 0.0
        return dot_product / (norm_emb1 * norm_emb2)

    def identify(self, face_image: np.ndarray, known_embeddings: Dict[str, np.ndarray]) -> Tuple[Optional[str], float]:
        """Identifies a face against known embeddings."""
        try:
            target_emb = self.extract_embedding(face_image)
        except ValueError:
            return None, 0.0

        best_match = None
        best_score = -1.0

        for name, emb in known_embeddings.items():
            score = self.compare_embeddings(target_emb, emb)
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= self.threshold:
            return best_match, best_score
        return None, best_score
