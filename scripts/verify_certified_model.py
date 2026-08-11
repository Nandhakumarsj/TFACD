"""Gate 5/6 gate: verify a certified checkpoint before runtime loading.

Exit code 0 iff every present integrity check passes - a missing signature is a
warning (unsigned certifications are still valid per Gate 5's "optional Ed25519
signature"), a manifest/signature mismatch is a hard failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tfacd.integrity.certification import verify_manifest
from tfacd.integrity.signing import verify_file

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--manifest", default=None)
parser.add_argument("--signature", default=None)
parser.add_argument("--public-key", default="artifacts/keys/certification_ed25519_public.pem")
args = parser.parse_args()

model_path = Path(args.model)
manifest_path = Path(args.manifest) if args.manifest else model_path.with_suffix(model_path.suffix + ".manifest.json")
signature_path = Path(args.signature) if args.signature else model_path.with_suffix(model_path.suffix + ".sig")

ok = True
if not manifest_path.exists():
    print(f"FAIL: no manifest at {manifest_path}")
    ok = False
elif not verify_manifest(model_path, manifest_path):
    print(f"FAIL: sha256 mismatch against {manifest_path}")
    ok = False
else:
    print("PASS: sha256 manifest verified")

if signature_path.exists():
    public_key_path = Path(args.public_key)
    if not public_key_path.exists():
        print(f"FAIL: signature present at {signature_path} but public key missing at {public_key_path}")
        ok = False
    elif verify_file(model_path, public_key_path, signature_path):
        print("PASS: Ed25519 signature verified")
    else:
        print(f"FAIL: signature verification failed against {signature_path}")
        ok = False
else:
    print(f"WARN: no signature found at {signature_path} (unsigned certification)")

sys.exit(0 if ok else 1)
