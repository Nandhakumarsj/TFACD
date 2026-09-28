# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✅ Active |
| Older tags / forks | ❌ No support |

Only the `main` branch receives security patches.

---

## Reporting a Vulnerability

**Please do not open a public GitHub Issue for security vulnerabilities.**

If you discover a security bug — including vulnerabilities in dependency
handling, model-integrity checks, the trust-boundary executor, or anything
that could allow an attacker to bypass the TFACD decision engine — please
report it privately using one of the following channels:

1. **GitHub Private Vulnerability Reporting** (preferred)  
   Go to the repository → **Security** tab → **"Report a vulnerability"**.

2. **Email**  
   Send details to the maintainer. You can find the contact address in the
   `[project]` section of `pyproject.toml` once a maintainer email is added,
   or via the GitHub profile linked to the repository.

### What to Include

Please provide as much detail as possible:

- A clear description of the vulnerability and its potential impact
- Steps to reproduce (config, command, expected vs actual output)
- Affected versions / branches
- Whether you have a proposed fix or patch

### Response Timeline

| Stage | Target |
|---|---|
| Acknowledgement | Within **72 hours** |
| Initial assessment | Within **7 days** |
| Fix or mitigation | Within **30 days** (complex issues may take longer) |
| Public disclosure | After fix is released and reporter notified |

We follow a **responsible-disclosure** policy: we will credit reporters in
the release notes unless they prefer to remain anonymous.

---

## Security Design Notes

TFACD enforces several layers of protection that contributors should be aware
of and must not weaken:

<!--
REVIEW FIX (Section 42 - stale documentation): this table previously
described an OLDER/aspirational architecture that doesn't match the current
code:
  - "SHA-256 hash pinned in configs/*.yaml" - no config file pins a model
    hash. The actual mechanism is manifest + signature (see
    src/tfacd/integrity/certification.py: write_manifest/verify_release),
    checked at load time by streaming/pipeline.py.
  - "Differential-privacy noise applied via dp_noise_scale" - this option
    does not exist anywhere in the codebase. There is currently no
    differential-privacy mechanism in the federated path. Removed rather
    than left in place describing a feature that isn't there.
  - "executors.py allow-list" - src/tfacd/trust_boundary/executors.py holds
    real execution DRIVERS (command/webhook/pluggable), not the allow-list
    itself. The actual capability whitelist enforcement lives in
    src/tfacd/trust_boundary/capability_enforcement.py, checked against
    configs/trust_policy.yaml's capability_whitelist and
    configs/threat_context.yaml's per-incident allowed_playbooks.
Keep this table in sync with the code it describes - a security doc that
describes a different system than what's running is worse than no doc.
-->

| Layer | Mechanism |
|---|---|
| **Model integrity** | Model release manifest (`sha256` + metadata + `status`) signed with Ed25519; verified via `integrity/certification.py::verify_release()` before every streaming inference run (`streaming/pipeline.py`) |
| **Trust boundary** | Every LLM/template-generated action plan passes through the `AdaptiveSemanticTrustBoundary` pipeline (preprocessing → dynamic trust scoring → capability whitelist + per-incident `allowed_playbooks` check in `capability_enforcement.py`) before any executor runs it |
| **Federated integrity** | `IntegrityAwareStrategy` (PCA + clustering-based outlier detection, EMA trust, robust trimmed-mean aggregation) rejects or down-weights suspicious client updates; a federation below `min_security_quorum` is refused rather than aggregated. There is currently no differential-privacy mechanism in the federated path |
| **Credential handling** | No API keys or `.pem` files committed; `.dockerignore` and `.gitignore` exclude them |
| **Dependency pinning** | `pyproject.toml` uses `>=`/`<` bounds; lock files used in Docker builds |
| **Audit trail** | Hash-chained, append-only audit log (`trust_boundary/audit.py`); a log that fails chain verification at startup is refused further writes rather than silently extended |

Known limitations (documented rather than hidden - see the code review this
project has been through for the full list): the certification public key is
currently read from a local `artifacts/keys` path rather than a
deployment-pinned trust root; the FTIL outlier detector's separation-quality
gate and quorum enforcement reduce but do not eliminate the risk of a
sufficiently large coordinated set of malicious clients; and per-client
update signing (as opposed to the final aggregated model's signature) is not
yet implemented.

Any change that weakens these controls requires explicit maintainer sign-off.
