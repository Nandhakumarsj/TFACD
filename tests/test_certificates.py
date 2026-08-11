from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.primitives.serialization import load_ssh_private_key, load_ssh_public_key

from tfacd.security.certificates import write_ca_and_server_cert, write_supernode_auth_keypair


def test_ca_and_server_cert_written_and_chained(tmp_path):
    paths = write_ca_and_server_cert(tmp_path)
    ca_cert = x509.load_pem_x509_certificate(paths["ca"].read_bytes())
    server_cert = x509.load_pem_x509_certificate(paths["server_cert"].read_bytes())

    assert server_cert.issuer == ca_cert.subject
    ca_cert.public_key().verify(
        server_cert.signature, server_cert.tbs_certificate_bytes, padding.PKCS1v15(), server_cert.signature_hash_algorithm
    )

    san = server_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "localhost" in san.get_values_for_type(x509.DNSName)


def test_supernode_auth_keypair_loads_as_ec_and_passes_flower_checks(tmp_path):
    paths = write_supernode_auth_keypair(tmp_path, "node-1")
    private_key = load_ssh_private_key(paths["private"].read_bytes(), password=None)
    public_key = load_ssh_public_key(paths["public"].read_bytes())

    # Matches the exact isinstance checks Flower's own CLI performs.
    assert isinstance(private_key, ec.EllipticCurvePrivateKey)
    assert isinstance(public_key, ec.EllipticCurvePublicKey)
    assert public_key.curve.name in ("secp384r1",)
