"""
Student enrollment module for EduVision AI.
Handles capturing face images, computing encrypted embeddings, and persisting to SQLite.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
from sqlalchemy.orm import Session as DBSession

from src.face_recognition.recognizer import FaceRecognizer
from src.security.encryption import EmbeddingEncryption
from src.database.models import Student, FaceEmbedding
from src.database.connection import SessionLocal


class EnrollmentManager:
    """
    Manages student face capture, embedding extraction, encryption, and database storage.
    """
    def __init__(self, data_dir: str = "data/enrolled_faces"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.recognizer = FaceRecognizer()
        self.crypto = EmbeddingEncryption()

    def enroll_from_images(
        self,
        student_id: int,
        images: List[np.ndarray],
        db: Optional[DBSession] = None
    ) -> np.ndarray:
        """
        Extracts embeddings from provided images, averages them into a canonical
        representation, encrypts it, and saves it in the database.
        """
        if not images:
            raise ValueError("At least one face image is required for enrollment.")

        embeddings = []
        for img in images:
            try:
                emb = self.recognizer.extract_embedding(img)
                embeddings.append(emb)
            except Exception as e:
                # Skip invalid or no-face frames
                continue

        if not embeddings:
            raise ValueError("No valid faces were detected in the provided enrollment images.")

        # Compute mean normalized embedding
        mean_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        # Encrypt the embedding for privacy
        encrypted_bytes = self.crypto.encrypt_embedding(mean_emb)

        # Persist to database
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            # Check if student exists
            student = db.query(Student).filter(Student.id == student_id).first()
            if not student:
                raise ValueError(f"Student with ID {student_id} does not exist.")

            # Replace or add face embedding
            existing = db.query(FaceEmbedding).filter(FaceEmbedding.student_id == student_id).first()
            if existing:
                existing.embedding_data = encrypted_bytes
            else:
                new_record = FaceEmbedding(
                    student_id=student_id,
                    embedding_data=encrypted_bytes
                )
                db.add(new_record)

            db.commit()
        finally:
            if close_db:
                db.close()

        # Save a reference photo to disk
        student_dir = self.data_dir / str(student_id)
        student_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(student_dir / "canonical.jpg"), images[0])

        return mean_emb

    def load_known_embeddings(self, db: Optional[DBSession] = None) -> Dict[int, np.ndarray]:
        """
        Loads and decrypts all enrolled student embeddings from the database.
        Returns a dictionary mapping {student_id: 128-d numpy array}.
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        known = {}
        try:
            records = db.query(FaceEmbedding).join(Student, FaceEmbedding.student_id == Student.id).filter(Student.is_active == True).all()
            for rec in records:
                try:
                    decrypted = self.crypto.decrypt_embedding(rec.embedding_data, shape=(128,))
                    known[rec.student_id] = decrypted
                except Exception as e:
                    print(f"Warning: Failed to decrypt embedding for student {rec.student_id}: {e}")
        finally:
            if close_db:
                db.close()

        return known
