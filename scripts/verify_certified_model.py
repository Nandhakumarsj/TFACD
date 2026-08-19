"""Gate 5/6 gate: verify a certified checkpoint before runtime loading.

Exit code 0 iff every check that ran passes. Thin CLI over
integrity/certification.py::verify_release - the same policy streaming/pipeline.py
uses before loading a checkpoint, so the two can't drift.
"""

from __future__ import annotations

import argparse
import sys

from tfacd.integrity.certification import verify_release

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--manifest", default=None)
parser.add_argument("--signature", default=None)
parser.add_argument("--public-key", default="artifacts/keys/certification_ed25519_public.pem")
parser.add_argument("--no-require-signature", action="store_true")
parser.add_argument("--no-require-certified-status", action="store_true")
args = parser.parse_args()

result = verify_release(
    args.model,
    manifest_path=args.manifest,
    signature_path=args.signature,
    public_key_path=args.public_key,
    require_signature=not args.no_require_signature,
    require_certified_status=not args.no_require_certified_status,
)

print(f"status:    {'PASS' if result.status_ok else 'FAIL'}")
print(f"sha256:    {'PASS' if result.sha256_ok else 'FAIL'}")
print(f"signature: {'PASS' if result.signature_ok else 'FAIL' if result.signature_ok is False else 'SKIPPED'}")
for reason in result.reasons:
    print(f"  - {reason}")
print("OVERALL:", "PASS" if result.ok else "FAIL")

sys.exit(0 if result.ok else 1)
