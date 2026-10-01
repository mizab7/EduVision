"""
Camera capture module.
Provides a CameraManager class to interact with cv2.VideoCapture.
"""

import time
import cv2
import numpy as np
from typing import Optional, Tuple


class CameraManager:
    """
    Manages camera capture and processing.
    """
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera with index {self.camera_index}")
        
        self.start_time = time.time()
        self.frame_count = 0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a frame from the camera."""
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame if ret else None

    def get_fps(self) -> float:
        """Calculates current FPS."""
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0.0
        return self.frame_count / elapsed_time

    def release(self):
        """Releases the camera resource."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def run_camera_preview():
    """Runs a live preview window showing the camera feed with FPS."""
    try:
        with CameraManager() as cam:
            while True:
                ret, frame = cam.read_frame()
                if not ret or frame is None:
                    print("Failed to read frame.")
                    break
                
                fps = cam.get_fps()
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Camera Preview", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_preview()
