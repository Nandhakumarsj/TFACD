from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(model_path: str | Path, metadata: dict, output_path: str | Path | None = None) -> Path:
    model = Path(model_path)
    out = Path(output_path) if output_path else model.with_suffix(model.suffix + ".manifest.json")
    payload = {"model": model.name, "sha256": sha256_file(model), "metadata": metadata}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def verify_manifest(model_path: str | Path, manifest_path: str | Path) -> bool:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return payload["sha256"] == sha256_file(model_path)


def chain_hash(previous_hash: str, payload: bytes) -> str:
    """Tamper-evident (not immutable) hash chaining for append-only logs: each
    entry's hash depends on the previous entry's hash, so altering or removing
    an entry breaks every hash after it.
    """
    digest = hashlib.sha256()
    digest.update(previous_hash.encode("utf-8"))
    digest.update(payload)
    return digest.hexdigest()


@dataclass
class ReleaseVerification:
    ok: bool
    status_ok: bool
    sha256_ok: bool
    signature_ok: bool | None  # None: no signature check was performed at all
    reasons: list[str] = field(default_factory=list)


def verify_release(
    model_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    signature_path: str | Path | None = None,
    public_key_path: str | Path = "artifacts/keys/certification_ed25519_public.pem",
    require_signature: bool = True,
    require_certified_status: bool = True,
) -> ReleaseVerification:
    """Shared verification policy for anything that loads a certified checkpoint
    (scripts/verify_certified_model.py, streaming/pipeline.py) - one place so the
    two can't drift. Check order matters: status first (fails fast, no key
    material touched, self-explanatory message), then sha256, then signature -
    the only one of the three that actually detects a retraining run silently
    replacing the certified model (every training run rewrites the manifest, so
    sha256 alone always "passes" against whatever the file currently is).
    """
    model = Path(model_path)
    manifest = Path(manifest_path) if manifest_path else model.with_suffix(model.suffix + ".manifest.json")
    signature = Path(signature_path) if signature_path else model.with_suffix(model.suffix + ".sig")
    reasons: list[str] = []

    if not manifest.exists():
        return ReleaseVerification(False, False, False, None, [f"no manifest at {manifest}"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    status = payload.get("metadata", {}).get("status")

    status_ok = (not require_certified_status) or status == "certified"
    if not status_ok:
        reasons.append(f"status={status!r}, expected 'certified' (run: python scripts/certify_model.py {model} --sign)")

    sha256_ok = payload.get("sha256") == sha256_file(model)
    if not sha256_ok:
        reasons.append(f"sha256 mismatch against {manifest}")

    signature_ok: bool | None = None
    if require_signature or signature.exists():
        from tfacd.integrity.signing import verify_file  # local import: avoids a hard cryptography dependency for callers that never touch signatures

        if not signature.exists():
            signature_ok = False
            reasons.append(f"no signature at {signature} (run: python scripts/certify_model.py {model} --sign)")
        elif not Path(public_key_path).exists():
            signature_ok = False
            reasons.append(f"signature present but public key missing at {public_key_path}")
        else:
            signature_ok = verify_file(model, public_key_path, signature)
            if not signature_ok:
                reasons.append(f"signature verification failed against {signature}")

    ok = status_ok and sha256_ok and signature_ok is not False
    return ReleaseVerification(ok, status_ok, sha256_ok, signature_ok, reasons)
