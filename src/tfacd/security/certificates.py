"""Certs/keys for Flower's actual deployment-mode security, not "mTLS".

flower-supernode's --root-certificates flag verifies the SERVER's cert (Flower's
own docstring: "This is NOT a client certificate for mTLS"). Mutual trust in
this Flower version is server-authenticated TLS (this module's CA/server cert)
plus a separate SuperNode public-key node-authentication mechanism (this
module's EC/OpenSSH keypairs) - not literal client-certificate mTLS.
"""

from __future__ import annotations

import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

_VALIDITY_DAYS = 365


def _self_signed_ca(common_name: str) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _server_cert(ca_key: rsa.RSAPrivateKey, ca_cert: x509.Certificate, common_name: str) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=_VALIDITY_DAYS))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    return key, cert


def write_ca_and_server_cert(output_dir: str | Path) -> dict[str, Path]:
    """Writes ca.pem (SuperLink's --ssl-ca-certfile), server.pem/server_key.pem
    (--ssl-certfile/--ssl-keyfile). SuperNode's --root-certificates uses ca.pem too."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ca_key, ca_cert = _self_signed_ca("TFACD Local CA")
    server_key, server_cert = _server_cert(ca_key, ca_cert, "127.0.0.1")

    paths = {"ca": out / "ca.pem", "server_cert": out / "server.pem", "server_key": out / "server_key.pem"}
    paths["ca"].write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    paths["server_cert"].write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    paths["server_key"].write_bytes(
        server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return paths


def generate_supernode_auth_keypair() -> tuple[bytes, bytes]:
    """EC P-384 keypair in OpenSSH format - SuperNode auth requires this exact
    shape (confirmed against flwr/supernode/cli/flower_supernode.py's
    load_ssh_private_key + isinstance(EllipticCurvePrivateKey) check, and
    flwr/cli/supernode/register.py's load_ssh_public_key + NIST-curve check),
    NOT the Ed25519/PKCS8 shape integrity/signing.py uses for model signing.
    """
    private_key = ec.generate_private_key(ec.SECP384R1())
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH
    )
    return private_bytes, public_bytes


def write_supernode_auth_keypair(output_dir: str | Path, node_name: str) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    private_bytes, public_bytes = generate_supernode_auth_keypair()
    paths = {"private": out / f"{node_name}_auth", "public": out / f"{node_name}_auth.pub"}
    paths["private"].write_bytes(private_bytes)
    paths["public"].write_bytes(public_bytes)
    return paths
