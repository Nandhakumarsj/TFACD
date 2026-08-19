from __future__ import annotations

import os
import stat
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def _restrict_to_owner(path: Path) -> None:
    """Best-effort owner-only permissions - see security/certificates.py's
    identical helper for why this is a no-op-beyond-read-only on Windows but
    real protection on this project's actual Linux IIoT deployment target."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def generate_keypair(private_path: str | Path, public_path: str | Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_path = Path(private_path)
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    _restrict_to_owner(private_path)
    Path(public_path).write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def sign_file(path: str | Path, private_key_path: str | Path, signature_path: str | Path) -> None:
    private_key = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("Expected an Ed25519 private key")
    Path(signature_path).write_bytes(private_key.sign(Path(path).read_bytes()))


def verify_file(path: str | Path, public_key_path: str | Path, signature_path: str | Path) -> bool:
    public_key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("Expected an Ed25519 public key")
    try:
        public_key.verify(Path(signature_path).read_bytes(), Path(path).read_bytes())
        return True
    except Exception:
        return False
