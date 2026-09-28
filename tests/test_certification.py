import json

from tfacd.integrity.certification import canonical_manifest_bytes, verify_release, write_manifest
from tfacd.integrity.signing import generate_keypair, sign_bytes


def _make_release(tmp_path, status="certified", sign=True):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake checkpoint bytes")
    manifest_path = write_manifest(model_path, {"status": status})

    private_key = public_key = None
    if sign:
        private_key, public_key = tmp_path / "priv.pem", tmp_path / "pub.pem"
        generate_keypair(private_key, public_key)
        # REVIEW FIX (Section 14): sign the canonical MANIFEST (model hash +
        # metadata + status together), matching scripts/certify_model.py -
        # not the model file's raw bytes.
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature_path = model_path.with_suffix(model_path.suffix + ".sig")
        signature_path.write_bytes(sign_bytes(canonical_manifest_bytes(manifest_payload), private_key))

    return model_path, public_key


def test_certified_and_signed_passes_every_check(tmp_path):
    model_path, public_key = _make_release(tmp_path)
    result = verify_release(model_path, public_key_path=public_key)
    assert result.ok
    assert result.status_ok and result.sha256_ok and result.signature_ok is True
    assert result.reasons == []


def test_sha256_mismatch_fails(tmp_path):
    model_path, public_key = _make_release(tmp_path)
    model_path.write_bytes(b"tampered bytes, different content")
    result = verify_release(model_path, public_key_path=public_key)
    assert not result.ok
    assert not result.sha256_ok
    assert any("sha256" in r for r in result.reasons)


def test_uncertified_status_fails_by_default(tmp_path):
    model_path, public_key = _make_release(tmp_path, status="trained-uncertified")
    result = verify_release(model_path, public_key_path=public_key)
    assert not result.ok
    assert not result.status_ok
    assert any("trained-uncertified" in r for r in result.reasons)


def test_uncertified_status_allowed_when_not_required(tmp_path):
    model_path, public_key = _make_release(tmp_path, status="trained-uncertified")
    result = verify_release(model_path, public_key_path=public_key, require_certified_status=False)
    assert result.status_ok
    assert result.ok


def test_missing_signature_fails_when_required(tmp_path):
    model_path, public_key = _make_release(tmp_path, sign=False)
    result = verify_release(model_path, public_key_path=tmp_path / "no_such_key.pem", require_signature=True)
    assert not result.ok
    assert result.signature_ok is False
    assert any("no signature" in r for r in result.reasons)


def test_missing_signature_allowed_when_not_required(tmp_path):
    model_path, public_key = _make_release(tmp_path, sign=False)
    result = verify_release(model_path, public_key_path=tmp_path / "no_such_key.pem", require_signature=False)
    assert result.ok
    assert result.signature_ok is None


def test_tampered_signature_fails(tmp_path):
    model_path, public_key = _make_release(tmp_path)
    # Re-sign a completely different manifest so the on-disk .sig no longer matches.
    other_manifest = {"model": "other.pt", "sha256": "0" * 64, "metadata": {"status": "certified"}}
    other_priv, other_pub = tmp_path / "other_priv.pem", tmp_path / "other_pub.pem"
    generate_keypair(other_priv, other_pub)
    signature_path = model_path.with_suffix(model_path.suffix + ".sig")
    signature_path.write_bytes(sign_bytes(canonical_manifest_bytes(other_manifest), other_priv))

    result = verify_release(model_path, public_key_path=public_key)
    assert not result.ok
    assert result.signature_ok is False
    assert any("manifest signature verification failed" in r for r in result.reasons)


def test_manifest_status_tampering_without_resigning_is_caught(tmp_path):
    """REVIEW FIX (Section 14 - certification signs the wrong artifact): this
    is the exact attack the fix closes. Previously the signature only covered
    the model FILE, so editing metadata.status in the manifest from
    "trained-uncertified" to "certified" - without re-signing anything - still
    passed verification, because the model bytes (the only thing signed)
    never changed. Now the signature covers the manifest itself, so this edit
    must be caught."""
    model_path, public_key = _make_release(tmp_path, status="trained-uncertified")
    manifest_path = model_path.with_suffix(model_path.suffix + ".manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metadata"]["status"] = "certified"  # forged, no re-signing
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    result = verify_release(model_path, public_key_path=public_key)
    assert not result.ok
    assert result.status_ok  # the forged status itself reads as "certified"...
    assert result.signature_ok is False  # ...but the manifest signature no longer matches.
    assert any("manifest signature verification failed" in r for r in result.reasons)


def test_missing_manifest_fails_cleanly(tmp_path):
    model_path = tmp_path / "no_manifest.pt"
    model_path.write_bytes(b"orphan checkpoint")
    result = verify_release(model_path)
    assert not result.ok
    assert any("no manifest" in r for r in result.reasons)
