"""
Student enrollment module for capturing and saving known face data.
"""

import os
import cv2
import numpy as np
from typing import List
from ..camera.capture import CameraManager
from .recognizer import FaceRecognizer

class EnrollmentManager:
    """
    Handles capturing, saving, and processing face enrollment data.
    """
    def __init__(self, data_dir: str = 'data/enrolled_faces'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.recognizer = FaceRecognizer()

    def capture_enrollment_images(self, camera: CameraManager, student_id: str, num_images: int = 5) -> List[np.ndarray]:
        """Captures multiple images for enrollment."""
        images = []
        print(f"Starting capture for {student_id}. Please look at the camera.")
        
        captured = 0
        while captured < num_images:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                continue
                
            cv2.imshow("Enrollment Capture - Press 'c' to capture", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                images.append(frame.copy())
                captured += 1
                print(f"Captured {captured}/{num_images}")
            elif key == ord('q'):
                break
                
        cv2.destroyAllWindows()
        return images

    def save_enrollment(self, student_id: str, images: List[np.ndarray]) -> None:
        """Saves enrollment images to disk."""
        student_dir = os.path.join(self.data_dir, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            filepath = os.path.join(student_dir, f"{i}.jpg")
            cv2.imwrite(filepath, img)
            
        print(f"Saved {len(images)} images for {student_id}")

    def generate_embeddings(self, student_id: str) -> np.ndarray:
        """Generates a mean embedding from saved enrollment images."""
        student_dir = os.path.join(self.data_dir, student_id)
        if not os.path.exists(student_dir):
            raise ValueError(f"No enrollment data found for {student_id}")
            
        embeddings = []
        for filename in os.listdir(student_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(student_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    try:
                        emb = self.recognizer.extract_embedding(img)
                        embeddings.append(emb)
                    except ValueError:
                        pass
                        
        if not embeddings:
            raise ValueError(f"Failed to extract embeddings for {student_id}")
            
        mean_embedding = np.mean(embeddings, axis=0)
        # Normalize the mean embedding
        return mean_embedding / np.linalg.norm(mean_embedding)
