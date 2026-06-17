"""Fernet-based symmetric encryption for audit content samples.

Key is a URL-safe base64-encoded 32-byte value (standard Fernet format).
Generate one with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Fail-closed: if no key is configured, ``enabled`` is False and the capture path
skips storage entirely — we never write plaintext.
"""

from __future__ import annotations

from cryptography.fernet import Fernet


class ContentCipher:
    def __init__(self, key: str | None) -> None:
        self._f = Fernet(key.encode()) if key else None

    @property
    def enabled(self) -> bool:
        return self._f is not None

    def encrypt(self, text: str) -> bytes:
        if self._f is None:
            raise RuntimeError("ContentCipher: no key configured — guard with .enabled first")
        return self._f.encrypt(text.encode())

    def decrypt(self, token: bytes) -> str:
        if self._f is None:
            raise RuntimeError("ContentCipher: no key configured")
        return self._f.decrypt(token).decode()
