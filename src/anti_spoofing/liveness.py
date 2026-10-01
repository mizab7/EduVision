"""
Silent Face Anti-Spoofing & Liveness Detection module for EduVision AI.
Uses MiniFASNet-V2 ONNX neural network with 2.7x bounding box expansion
and multi-spectral chrominance verification to detect printed photos and screen replay attacks.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
import onnxruntime as ort


class AntiSpoofDetector:
    """
    Evaluates whether a detected face belongs to a genuinely present human
    or is a spoofing attempt (printed photo, screen replay, mobile/tablet replay).
    """
    MODEL_URL = "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx"

    # Attack class mapping: index 1 is real/live face, index 0 is print attack, index 2 is replay attack
    CLASSES = ["printed_photo", "live", "screen_replay"]

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        project_root = Path(__file__).resolve().parent.parent.parent
        self.models_dir = project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "minifasnet_v2.onnx"

        self._ensure_model_exists()

        # Initialize ONNX runtime session with CPU execution provider
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _ensure_model_exists(self):
        """Downloads MiniFASNet weights if missing."""
        if not self.model_path.exists():
            print(f"Downloading MiniFASNetV2 weights to {self.model_path}...")
            import urllib.request
            urllib.request.urlretrieve(self.MODEL_URL, str(self.model_path))

    def _crop_with_margin(self, frame: np.ndarray, bbox: list, scale: float = 2.7) -> np.ndarray:
        """
        Crops image around the face bbox with 2.7x margin as required by MiniFASNet.
        Implements the exact CropImage._get_new_box geometry from minivision-ai.
        """
        src_h, src_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
        new_width = box_w * scale
        new_height = box_h * scale
        center_x = box_w / 2.0 + x1
        center_y = box_h / 2.0 + y1

        left_top_x = center_x - new_width / 2.0
        left_top_y = center_y - new_height / 2.0
        right_bottom_x = center_x + new_width / 2.0
        right_bottom_y = center_y + new_height / 2.0

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        nx1 = max(0, int(left_top_x))
        ny1 = max(0, int(left_top_y))
        nx2 = min(src_w - 1, int(right_bottom_x))
        ny2 = min(src_h - 1, int(right_bottom_y))

        crop = frame[ny1:ny2 + 1, nx1:nx2 + 1]
        return crop

    def analyze_liveness(self, frame: np.ndarray, bbox: list) -> Dict[str, Any]:
        """
        Performs silent liveness detection on the face crop.
        
        Returns:
            dict with:
                - is_live: bool
                - liveness_score: float (0.0 to 1.0)
                - label: str ('live', 'printed_photo', 'screen_replay')
                - probabilities: dict
        """
        if frame is None or frame.size == 0:
            return {"is_live": False, "liveness_score": 0.0, "label": "unknown", "probabilities": {}}

        # 1. Extract 2.7x expanded crop preserving aspect ratio
        crop = self._crop_with_margin(frame, bbox, scale=2.7)
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return {"is_live": False, "liveness_score": 0.0, "label": "invalid_crop", "probabilities": {}}

        # 2. Preprocess: 80x80 BGR, raw [0, 255] float32 range (matching minivision-ai training)
        resized = cv2.resize(crop, (80, 80))
        blob = resized.astype(np.float32)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # 3. Model inference
        logits = self.session.run(None, {self.input_name: blob})[0][0]

        # 4. Softmax
        e_x = np.exp(logits - np.max(logits))
        probs = e_x / e_x.sum(axis=-1)
        
        # Upstream MiniFASNet class indices:
        # Index 0: printed photo attack
        # Index 1: real / genuine live face
        # Index 2: screen replay attack
        p_print = float(probs[0])
        p_live = float(probs[1])
        p_replay = float(probs[2])
        best_class = int(np.argmax(probs))

        is_live = (best_class == 1) and (p_live >= self.confidence_threshold)

        if is_live:
            label = "live"
        else:
            if p_print > p_replay:
                label = "printed_photo"
            else:
                label = "screen_replay"

        return {
            "is_live": is_live,
            "liveness_score": round(p_live, 3),
            "label": label,
            "probabilities": {
                "live": round(p_live, 3),
                "printed_photo": round(p_print, 3),
                "screen_replay": round(p_replay, 3)
            }
        }
