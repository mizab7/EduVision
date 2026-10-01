"""
Security & Encryption module for EduVision AI.
Provides Fernet AES-128-CBC encryption for facial embeddings to preserve student privacy.
"""

import base64
import hashlib
from typing import Optional
import numpy as np
from cryptography.fernet import Fernet
from config.settings import get_settings


class EmbeddingEncryption:
    """
    Encrypts and decrypts numpy facial embeddings using AES Fernet.
    """
    def __init__(self, secret_key: Optional[str] = None):
        settings = get_settings()
        raw_key = secret_key or settings.SECRET_KEY or "eduvision_ai_secure_master_key_2026"
        # Generate 32-byte URL-safe base64 key using SHA-256
        key_bytes = hashlib.sha256(raw_key.encode("utf-8")).digest()
        self.fernet_key = base64.urlsafe_b64encode(key_bytes)
        self.cipher = Fernet(self.fernet_key)

    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """Serializes and encrypts a 1D or 2D float32 numpy embedding."""
        raw_bytes = np.ascontiguousarray(embedding, dtype=np.float32).tobytes()
        return self.cipher.encrypt(raw_bytes)

    def decrypt_embedding(self, encrypted_bytes: bytes, shape: tuple = (128,)) -> np.ndarray:
        """Decrypts and recovers the numpy embedding."""
        decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
        arr = np.frombuffer(decrypted_bytes, dtype=np.float32)
        if shape:
            arr = arr.reshape(shape)
        return arr
