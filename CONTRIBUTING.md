# Contributing to TFACD

Thank you for taking the time to contribute! This document explains the
process for reporting bugs, proposing enhancements, and submitting pull
requests to **TFACD – Trustworthy Federated Agentic Cyber Defense for IIoT**.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [I Have a Question](#i-have-a-question)
3. [Reporting Bugs](#reporting-bugs)
4. [Suggesting Enhancements](#suggesting-enhancements)
5. [Your First Code Contribution](#your-first-code-contribution)
6. [Pull Request Process](#pull-request-process)
7. [Development Setup](#development-setup)
8. [Coding Standards](#coding-standards)
9. [Commit Message Convention](#commit-message-convention)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating you agree to uphold its standards. Please report unacceptable
behaviour to the maintainers via GitHub Issues.

---

## I Have a Question

Before opening a new issue:
- Check [existing Issues](https://github.com/Nandhakumarsj/TFACD/issues)
- Read the [README](README.md) and [DOCKER_GUIDE](DOCKER_GUIDE.md)
- Look at the inline docstrings and config comments in `configs/`

If you still need help, open a **Discussion** (not an Issue) labelled
`question`.

---

## Reporting Bugs

### Before Submitting

1. Make sure you are on the **latest `main`**.
2. Check if the bug is already tracked in [Issues](https://github.com/Nandhakumarsj/TFACD/issues).
3. For security vulnerabilities, **do not open a public issue** — see [SECURITY.md](SECURITY.md).

### How to Report

Open a new Issue using the **Bug Report** template and include:

| Field | What to include |
|---|---|
| **Environment** | OS, Python version, GPU / CUDA version, `pip show tfacd` output |
| **Steps to reproduce** | Minimal commands that trigger the bug |
| **Expected behaviour** | What should have happened |
| **Actual behaviour** | What actually happened, with full traceback |
| **Config snapshot** | Relevant section of your `.yaml` config (no credentials) |

---

## Suggesting Enhancements

Open a new Issue using the **Feature Request** template and describe:

- The problem you are solving and its motivation
- The proposed solution, with pseudocode or an example if possible
- Alternatives you considered
- Whether this requires a new optional dependency (and why it can't be optional)

---

## Your First Code Contribution

1. **Fork** the repository and clone your fork.
2. Create a new branch: `git checkout -b feat/my-feature` or `fix/the-bug`.
3. Make your changes (see [Development Setup](#development-setup)).
4. Run the test suite and linter (see [Coding Standards](#coding-standards)).
5. Open a pull request against `main`.

We label beginner-friendly tasks with **`good first issue`**. Filter the
Issues list by that label to find a starting point.

---

## Pull Request Process

1. **One logical change per PR** — keep PRs small and focused.
2. All CI checks must pass (ruff, mypy, pytest).
3. New behaviour must be covered by at least one test in `tests/`.
4. Update the relevant section in `README.md` or `docs/` if the public API
   or CLI changes.
5. Two approvals are required from maintainers before merging.
6. PRs are merged using **Squash & Merge** to keep the history clean.

---

## Development Setup

```bash
# 1. Clone your fork
git clone https://github.com/<your-handle>/TFACD.git && cd TFACD

# 2. Create the virtual environment and install all dev dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,flower,agentic-llm,viz,explainability,mqtt]"

# 3. Verify the install
python -m pytest tests/ -q

# 4. Run the linter
ruff check src/ tests/ scripts/
mypy src/tfacd --ignore-missing-imports
```

> **GPU note**: For CUDA 12.6 PyTorch wheels, replace the `torch` install
> with the index URL documented in [DOCKER_GUIDE.md](DOCKER_GUIDE.md).

---

## Coding Standards

| Tool | Purpose | Config |
|---|---|---|
| **ruff** | Linting + auto-format | `pyproject.toml [tool.ruff]` |
| **mypy** | Static type checking | inline |
| **pytest** | Unit & integration tests | `tests/` |

Rules:
- Line length: **100 characters**
- All public functions and classes must have docstrings.
- Avoid hard-coded paths — use `configs/*.yaml` and `pathlib.Path`.
- New optional dependencies must be guarded with `try/except ImportError`.
- Do not commit datasets, model weights, or `.pem` keys.

---

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

**Examples**:
```
feat(llm): add gemma4:26b as primary decision engine
fix(ftil): correct cosine similarity aggregation normalisation
docs(readme): add docker quick-start section
test(temporal): add unit tests for EWMA drift scorer
```
