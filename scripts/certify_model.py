from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfacd.integrity.certification import verify_manifest, write_manifest
from tfacd.integrity.signing import generate_keypair, sign_file, verify_file

parser = argparse.ArgumentParser(description="Gate 5: certify a federated checkpoint for runtime release.")
parser.add_argument("model", help="Path to the model checkpoint (.pt)")
parser.add_argument("--status", default="certified")
parser.add_argument("--sign", action="store_true", help="Also Ed25519-sign the checkpoint")
parser.add_argument("--key-dir", default="artifacts/keys")
args = parser.parse_args()

model_path = Path(args.model)
metadata: dict = {"status": args.status}

# Fold in what was already recorded (training strategy, proximal_mu, round count)
# rather than discarding it - write_manifest below overwrites the same
# <model>.manifest.json path the training run wrote to.
existing_manifest = model_path.with_suffix(model_path.suffix + ".manifest.json")
if existing_manifest.exists():
    metadata["training_metadata"] = json.loads(existing_manifest.read_text(encoding="utf-8")).get("metadata", {})

trust_log = Path("artifacts/models/ftil_trust_log.jsonl")
if trust_log.exists():
    rounds = [json.loads(line) for line in trust_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata["ftil_trust_summary"] = {
        "rounds_logged": len(rounds),
        "total_rejected_validation": sum(len(r["rejected_validation"]) for r in rounds),
        "total_rejected_detector": sum(len(r["rejected_detector"]) for r in rounds),
    }

manifest_path = write_manifest(model_path, metadata)
print(f"manifest: {manifest_path}")
assert verify_manifest(model_path, manifest_path)
print("sha256 verification: PASS")

if args.sign:
    key_dir = Path(args.key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    private_key = key_dir / "certification_ed25519.pem"
    public_key = key_dir / "certification_ed25519_public.pem"
    if not private_key.exists():
        generate_keypair(private_key, public_key)
        print(f"generated new keypair: {private_key}, {public_key}")
    signature_path = model_path.with_suffix(model_path.suffix + ".sig")
    sign_file(model_path, private_key, signature_path)
    print(f"signature: {signature_path}")
    assert verify_file(model_path, public_key, signature_path)
    print("signature verification: PASS")

print(f"\nCertified: {model_path}")
