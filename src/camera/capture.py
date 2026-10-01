"""
Camera capture module for EduVision AI.
Provides DualCameraWorker with persistent pre-warmed dual hardware handles
(MacBook FaceTime HD & External 1080p Web Camera).
Instantaneous zero-latency switching with zero black frames.
"""

import time
import threading
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np


def enhance_low_light(frame: np.ndarray, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
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


class CameraWorker:
    """
    Maintains persistent hardware handles for all available cameras simultaneously.
    Switches between cameras instantaneously with zero warmup delay or black screen.
    """
    _instance = None
    _singleton_lock = threading.Lock()

    @classmethod
    def get_instance(cls, default_index: int = 0):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = CameraWorker(default_index=default_index)
            return cls._instance

    def __init__(self, default_index: int = 0):
        self.active_index = default_index
        self.low_light_boost = False
        self.running = True
        self.frame_lock = threading.Lock()
        
        self.latest_frame = None
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = time.time()
        
        # Pre-open both cameras simultaneously
        self.caps = {}
        self._init_hardware_devices()
        
        # Start background reader thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _init_hardware_devices(self):
        """Initializes and pre-warms all detected cameras."""
        for idx in [0, 1]:
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    # Warm up sensor auto-exposure
                    for _ in range(4):
                        cap.read()
                    self.caps[idx] = cap
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✅ CameraWorker pre-warmed Camera {idx} ({w}x{h})")
            except Exception as e:
                print(f"Warning: Failed to initialize camera {idx}: {e}")

    def switch_camera(self, new_index: int, low_light: bool = False):
        """
        Instantaneous zero-millisecond switch to target camera.
        No hardware teardown or reconnection needed.
        """
        self.low_light_boost = low_light
        if new_index in self.caps:
            self.active_index = new_index

    def set_low_light_boost(self, enable: bool):
        self.low_light_boost = enable

    def _capture_loop(self):
        """Background frame acquisition loop."""
        while self.running:
            cap = self.caps.get(self.active_index)
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Apply low-light boost if active
                    if self.low_light_boost:
                        frame = enhance_low_light(frame, clip_limit=3.0)
                    
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.frame_count += 1
                        now = time.time()
                        if now - self.fps_timer >= 1.0:
                            self.fps = self.frame_count / (now - self.fps_timer)
                            self.frame_count = 0
                            self.fps_timer = now
            time.sleep(0.015)

    def get_frame(self) -> Optional[np.ndarray]:
        """Returns a thread-safe copy of the latest captured frame."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_status(self) -> Dict[str, Any]:
        """Returns the current camera configuration and performance metrics."""
        with self.frame_lock:
            w, h = 0, 0
            if self.latest_frame is not None:
                h, w = self.latest_frame.shape[:2]
            return {
                "active_index": self.active_index,
                "resolution": f"{w}x{h}",
                "fps": round(self.fps, 1),
                "low_light_boost": self.low_light_boost,
                "is_streaming": self.latest_frame is not None
            }

    def release(self):
        """Stops background loop and releases all hardware handles."""
        self.running = False
        time.sleep(0.1)
        for idx, cap in self.caps.items():
            try:
                if cap is not None and cap.isOpened():
                    cap.release()
            except Exception:
                pass
        self.caps.clear()

    @classmethod
    def list_available_cameras(cls, max_tested: int = 2) -> List[Dict[str, Any]]:
        """Returns pre-configured human-friendly camera devices list."""
        return [
            {
                "index": 0,
                "label": "Camera 0 — Built-in Mac Camera",
                "resolution": "640x480",
                "is_active": True
            },
            {
                "index": 1,
                "label": "Camera 1 — External 1080p Web Camera (W100)",
                "resolution": "1920x1080",
                "is_active": True
            }
        ]


# Alias for backward compatibility
CameraManager = CameraWorker
