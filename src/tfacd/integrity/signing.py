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


def sign_bytes(data: bytes, private_key_path: str | Path) -> bytes:
    private_key = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("Expected an Ed25519 private key")
    return private_key.sign(data)


def verify_bytes(data: bytes, public_key_path: str | Path, signature: bytes) -> bool:
    public_key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("Expected an Ed25519 public key")
    try:
        public_key.verify(signature, data)
        return True
    except Exception:
        return False


def sign_file(path: str | Path, private_key_path: str | Path, signature_path: str | Path) -> None:
    """Signs the raw bytes of `path`. NOTE: for model-certification specifically,
    scripts/certify_model.py and certification.verify_release() sign/verify the
    CANONICAL MANIFEST (certification.canonical_manifest_bytes), not the model
    file directly - see certification.py module docstring for why. This
    function remains available (and is exercised directly by
    tests/test_signing.py) for signing an arbitrary file's raw bytes.
    """
    Path(signature_path).write_bytes(sign_bytes(Path(path).read_bytes(), private_key_path))


def verify_file(path: str | Path, public_key_path: str | Path, signature_path: str | Path) -> bool:
    return verify_bytes(Path(path).read_bytes(), public_key_path, Path(signature_path).read_bytes())
