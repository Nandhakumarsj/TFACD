from __future__ import annotations

import hashlib
import json
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
