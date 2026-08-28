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

| Layer | Mechanism |
|---|---|
| **Model integrity** | SHA-256 hash pinned in `configs/*.yaml`; verified before every inference run |
| **Trust boundary** | All LLM-generated actions pass through `executors.py` allow-list before execution |
| **Federated privacy** | Differential-privacy noise applied via `dp_noise_scale` config option |
| **Credential handling** | No API keys or `.pem` files committed; `.dockerignore` and `.gitignore` exclude them |
| **Dependency pinning** | `pyproject.toml` uses `>=`/`<` bounds; lock files used in Docker builds |

Any change that weakens these controls requires explicit maintainer sign-off.
