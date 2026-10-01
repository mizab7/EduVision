"""
Camera capture module for EduVision AI.
Provides CameraManager with high-definition capture, low-light enhancement (CLAHE),
and dynamic device switching.
"""

import time
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np


def enhance_low_light(frame: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the Luminance (L) channel in LAB color space to brighten dark/dim frames
    while preserving natural colors and facial details.
    """
    if frame is None or frame.size == 0:
        return frame
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


class CameraManager:
    """
    Manages camera capture, resolution configuration, and image processing filters.
    """
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        low_light_boost: bool = False
    ):
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height
        self.low_light_boost = low_light_boost

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera with index {self.camera_index}")

        # Request HD resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        self.start_time = time.time()
        self.frame_count = 0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a frame and optionally applies low-light enhancement."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        self.frame_count += 1

        if self.low_light_boost:
            frame = enhance_low_light(frame)

        return True, frame

    def get_fps(self) -> float:
        """Calculates current capture FPS."""
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0.0
        return self.frame_count / elapsed_time

    def get_resolution(self) -> Tuple[int, int]:
        """Returns the actual resolution of the camera stream."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        """Releases the camera hardware resource."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @staticmethod
    def list_available_cameras(max_tested: int = 4) -> List[Dict[str, Any]]:
        """
        Scans indexes 0..max_tested to detect available hardware video devices.
        """
        devices = []
        for idx in range(max_tested):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    label = f"Camera {idx}"
                    if idx == 0:
                        label += " (MacBook FaceTime HD)"
                    elif idx == 1:
                        label += " (External Web Camera)"
                    
                    devices.append({
                        "index": idx,
                        "label": label,
                        "resolution": f"{w}x{h}",
                        "is_active": True
                    })
                cap.release()
        return devices


def run_camera_preview(camera_index: int = 0, low_light_boost: bool = False):
    """Runs a live preview window showing the camera feed with FPS."""
    try:
        with CameraManager(camera_index=camera_index, low_light_boost=low_light_boost) as cam:
            w, h = cam.get_resolution()
            print(f"Opened Camera {camera_index} at {w}x{h}, Low-Light Boost={low_light_boost}")
            while True:
                ret, frame = cam.read_frame()
                if not ret or frame is None:
                    print("Failed to read frame.")
                    break
                
                fps = cam.get_fps()
                cv2.putText(frame, f"FPS: {fps:.1f} | Res: {w}x{h}", (15, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow("Camera Preview (Press 'q' to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_preview()
