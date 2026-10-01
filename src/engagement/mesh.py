"""
High-performance Face Mesh and 478 Facial Landmark Extractor for EduVision AI.
Uses MediaPipe FaceMesh V2 converted to ONNX Runtime (CPUExecutionProvider).
Runs safely without MediaPipe C++ Metal shader crashes on Apple Silicon (M1/M2/M3).
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import onnxruntime as ort


class FaceMeshExtractor:
    """
    Extracts 478 detailed 3D facial landmarks including iris centers,
    eye contours, and lip contours from face bounding boxes.
    """
    MODEL_URL = "https://huggingface.co/naklitechie/face-landmarks-onnx/resolve/main/face_landmarks.onnx"

    # Landmark indices matching MediaPipe 478-point topology
    # Eyes:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]      # [outer, top1, top2, inner, bottom2, bottom1]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]   # [outer, top1, top2, inner, bottom2, bottom1]
    LEFT_IRIS = 468                               # Center of left iris
    RIGHT_IRIS = 473                              # Center of right iris
    LEFT_IRIS_CONTOUR = [468, 469, 470, 471, 472]
    RIGHT_IRIS_CONTOUR = [473, 474, 475, 476, 477]

    # Mouth (for MAR / Yawn):
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308

    # Key points for solvePnP Head Pose:
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_CORNER = 33
    RIGHT_EYE_CORNER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.models_dir = project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "face_landmarks.onnx"

        self._ensure_model_exists()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _ensure_model_exists(self):
        if not self.model_path.exists():
            print(f"Downloading FaceMesh ONNX weights to {self.model_path}...")
            import urllib.request
            urllib.request.urlretrieve(self.MODEL_URL, str(self.model_path))

    def extract_landmarks(self, frame: np.ndarray, bbox: list) -> Optional[Dict[str, Any]]:
        """
        Extracts 478 3D facial landmarks for the given face bounding box.
        
        Args:
            frame: Full BGR frame (H, W, 3)
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Dictionary with:
                - landmarks: (478, 3) numpy array in frame pixel coordinates
                - norm_landmarks: (478, 3) numpy array normalized [0, 1] relative to face crop
                - presence: float (face presence score)
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # Expand crop with 1.25x margin to include chin, temples, and forehead
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0
        size = max(bw, bh) * 1.25

        crop_x1 = max(0, int(cx - size / 2.0))
        crop_y1 = max(0, int(cy - size / 2.0))
        crop_x2 = min(w, int(cx + size / 2.0))
        crop_y2 = min(h, int(cy + size / 2.0))

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        # Preprocess for ONNX: 256x256 RGB float32 [0..1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (256, 256))
        blob = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        outputs = self.session.run(None, {self.input_name: blob})
        raw_lmks = outputs[0].reshape(-1, 3)  # (478, 3) in [0..256] space
        presence = float(outputs[1].squeeze()) if len(outputs) > 1 else 1.0

        # Transform landmarks from 256x256 crop space back to original frame coordinates
        cw = crop_x2 - crop_x1
        ch = crop_y2 - crop_y1

        norm_lmks = raw_lmks / 256.0
        frame_lmks = np.zeros_like(raw_lmks)
        frame_lmks[:, 0] = crop_x1 + norm_lmks[:, 0] * cw
        frame_lmks[:, 1] = crop_y1 + norm_lmks[:, 1] * ch
        frame_lmks[:, 2] = norm_lmks[:, 2] * max(cw, ch)

        return {
            "landmarks": frame_lmks,
            "norm_landmarks": norm_lmks,
            "presence": presence,
            "crop_box": [crop_x1, crop_y1, crop_x2, crop_y2]
        }
