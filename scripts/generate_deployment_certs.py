from __future__ import annotations

import argparse
from pathlib import Path

from tfacd.security.certificates import write_ca_and_server_cert, write_supernode_auth_keypair

parser = argparse.ArgumentParser(description="Generate local CA/server TLS cert + SuperNode auth keypairs for deployment-mode Flower.")
parser.add_argument("--output-dir", default="artifacts/certs")
parser.add_argument("--num-supernodes", type=int, default=2)
args = parser.parse_args()

out = Path(args.output_dir)
cert_paths = write_ca_and_server_cert(out)
print("CA/server TLS cert (server-authenticated channel, not client-cert mTLS):")
for name, path in cert_paths.items():
    print(f"  {name}: {path}")

node_dir = out / "supernodes"
print(f"\nSuperNode auth keypairs (EC/OpenSSH, separate mechanism from the TLS cert above):")
for i in range(args.num_supernodes):
    node_name = f"supernode-{i}"
    node_paths = write_supernode_auth_keypair(node_dir, node_name)
    print(f"  {node_name}: {node_paths['private']} / {node_paths['public']}")
