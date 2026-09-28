from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

# REVIEW FIX (Section 14 - certification signs the wrong artifact): this used
# to sign/verify the MODEL FILE's raw bytes (sign_file/verify_file on
# model_path). The certified/uncertified STATUS lives only in the manifest
# JSON alongside it, which was never covered by that signature - an attacker
# able to edit the manifest (but not re-sign anything) could flip
# metadata.status from "trained-uncertified" to "certified" and the model
# signature would still verify, because it never said anything about status
# in the first place. The fix: sign the CANONICAL MANIFEST bytes (which
# already contain the model's sha256 + metadata + status together), and
# verify that same manifest signature plus a fresh sha256 recompute of the
# model file. That ties status, metadata, and model integrity into one
# signed artifact instead of two independently-trusted ones.
#
# Outstanding limitation this does NOT fix (flagged, not silently left
# unaddressed): the public key is still read from a local artifacts/keys
# path rather than a deployment-installed/pinned trust root. That is a
# deployment/infrastructure decision (where the trusted public key actually
# lives, how it's provisioned to verifying hosts) that depends on how this
# project is deployed - not something to invent a mechanism for without
# knowing that. Left as an explicit follow-up for the person building this.


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_bytes(payload: dict) -> bytes:
    """Deterministic byte encoding of a manifest payload (the exact dict shape
    write_manifest() produces: {"model", "sha256", "metadata"}), used as the
    thing that actually gets signed/verified for certification. Sorted keys +
    fixed separators so the same logical manifest always encodes identically
    regardless of dict insertion order.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


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
        from tfacd.integrity.signing import verify_bytes  # local import: avoids a hard cryptography dependency for callers that never touch signatures

        if not signature.exists():
            signature_ok = False
            reasons.append(f"no signature at {signature} (run: python scripts/certify_model.py {model} --sign)")
        elif not Path(public_key_path).exists():
            signature_ok = False
            reasons.append(f"signature present but public key missing at {public_key_path}")
        else:
            # REVIEW FIX (Section 14): verify against the CANONICAL MANIFEST
            # bytes (model hash + metadata + status together), not the model
            # file's raw bytes - see module docstring. sha256_ok above already
            # independently confirms the manifest's claimed hash matches the
            # actual model file, so together these two checks mean: the model
            # matches what the manifest claims, AND the manifest (hash,
            # metadata, status - all of it) matches what was actually signed.
            signature_ok = verify_bytes(canonical_manifest_bytes(payload), public_key_path, signature.read_bytes())
            if not signature_ok:
                reasons.append(f"manifest signature verification failed against {signature}")

    # signature_ok is None only when no check was performed at all (possible
    # only when require_signature=False AND no .sig file exists - see the
    # `if require_signature or signature.exists():` branch above). Spelled out
    # explicitly here (rather than the previous `signature_ok is not False`,
    # which passed None regardless of require_signature) so this stays correct
    # even if that branch's logic changes later, instead of depending on an
    # invariant the boolean check itself doesn't express.
    signature_component_ok = signature_ok is True or (signature_ok is None and not require_signature)
    ok = status_ok and sha256_ok and signature_component_ok
    return ReleaseVerification(ok, status_ok, sha256_ok, signature_ok, reasons)
