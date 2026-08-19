from tfacd.integrity.certification import verify_release, write_manifest
from tfacd.integrity.signing import generate_keypair, sign_file


def _make_release(tmp_path, status="certified", sign=True):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake checkpoint bytes")
    write_manifest(model_path, {"status": status})

    private_key = public_key = signature_path = None
    if sign:
        private_key, public_key = tmp_path / "priv.pem", tmp_path / "pub.pem"
        generate_keypair(private_key, public_key)
        signature_path = model_path.with_suffix(model_path.suffix + ".sig")
        sign_file(model_path, private_key, signature_path)

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
    # Re-sign a different (unrelated) model so the on-disk .sig no longer matches.
    other = tmp_path / "other.pt"
    other.write_bytes(b"a completely different checkpoint")
    other_priv, other_pub = tmp_path / "other_priv.pem", tmp_path / "other_pub.pem"
    generate_keypair(other_priv, other_pub)
    sign_file(other, other_priv, model_path.with_suffix(model_path.suffix + ".sig"))

    result = verify_release(model_path, public_key_path=public_key)
    assert not result.ok
    assert result.signature_ok is False
    assert any("signature verification failed" in r for r in result.reasons)


def test_missing_manifest_fails_cleanly(tmp_path):
    model_path = tmp_path / "no_manifest.pt"
    model_path.write_bytes(b"orphan checkpoint")
    result = verify_release(model_path)
    assert not result.ok
    assert any("no manifest" in r for r in result.reasons)
