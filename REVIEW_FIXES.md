# Review fixes — changelog

This documents every change made in response to the code review shared
alongside this codebase. It follows the review's own recommended
"correction order" (its Section 46, items A–H): fix the core seams before
adding any new features. Nothing outside that list (plus two clearly-scoped
bonus P1 fixes noted at the end) was touched, so anything not mentioned here
is unchanged from what you handed over.

Every change below also has an inline `REVIEW FIX (...)` comment at the
exact line(s) it touches, explaining the bug and the reasoning for the fix
chosen — this file is the index; the comments are the detail.

## Already fixed before this pass (verified, not re-touched)

**Item A — training/runtime `sequence_length` contract.** The review found
`configs/edge_iiot.yaml` defaulting to `sequence_length: 16` while
`streaming/pipeline.py` hard-rejects anything but `1`. That contradiction was
already resolved in the codebase you handed over (config already says `1`),
so nothing needed changing here.

## B — Temporal split leakage fallback (P0)

**File:** `src/tfacd/data/preprocess.py`

The session-safe temporal split silently fell back to a random
sequence-level `train_test_split` whenever it couldn't populate all three
splits. That fallback does not keep a session in a single split, so
overlapping windows from the same session could land on both sides of
train/test — reintroducing exactly the leakage the session-safe splitter
exists to prevent.

**Fix:** raise a `ValueError` with a diagnostic message (session counts,
what to try) instead of silently degrading to a leaky split. Added
`tests/test_data_preprocess.py::test_temporal_fallback_raises_instead_of_leaking`.

## C — Federated client identity spoofing (P0/P1)

**File:** `src/tfacd/federated/integrity_strategy.py`

`client_id` was read from client-supplied `content["client-metadata"]`
whenever present — i.e. the server trusted whatever ID a client claimed to
be. A malicious client could impersonate another client's ID and pollute its
EMA trust history.

**Fix:** `client_id` now comes only from `msg.metadata.src_node_id`, Flower's
authenticated sender identity. Client-supplied metadata is no longer read
for identity purposes at all. Added
`tests/test_integrity_strategy.py::test_client_metadata_cannot_spoof_identity`
and updated the test fixtures (`FakeReply`/`make_reply`) to carry an
authenticated `src_node_id` distinct from claimed metadata.

## Security quorum bypass (P0/P1, found alongside C and D)

**Files:** `src/tfacd/federated/integrity_strategy.py`,
`src/tfacd/federated/server_app.py`, `configs/edge_iiot.yaml`

`server_app.py`'s `min_train_nodes`/`min_available_nodes` can be as low as
2 — below the detector's own "`n_clients < 3` → accept all" threshold *and*
below the point where `trim_ratio` trims anything (`floor(2 * 0.2) == 0`). A
shrunk federation could silently bypass both the outlier detector and the
robust aggregator at once.

**Fix:** added `min_security_quorum` (default 3) to `IntegrityAwareStrategy`,
enforced independently of the detector, before it ever runs. Below quorum,
the round returns no aggregate at all (`ftil_insufficient_security_quorum`
metric set) instead of aggregating with "accept everyone" semantics. Wired
through `server_app.py` from `configs/edge_iiot.yaml`'s new
`integrity.min_security_quorum` key. Added
`tests/test_integrity_strategy.py::test_insufficient_security_quorum_skips_round`.

## D — Forced two-cluster problem (RED)

**File:** `src/tfacd/integrity/detector.py`

`AgglomerativeClustering(n_clusters=2)` was forced on every round with ≥3
clients, even when the client population had no real separation (e.g. 5
honest clients with near-identical updates still got split into "majority
benign / minority suspicious"). An honest client repeatedly landing in the
minority could eventually be rejected by EMA trust alone.

**Fix:** added `min_silhouette_for_rejection` (default 0.15, a conventional
"no substantial structure" floor per Kaufman & Rousseeuw's silhouette
interpretation guidelines). When the cohort is degenerate, or silhouette is
undefined or below this floor, every client is accepted for that round's
clustering signal instead of an arbitrary split being trusted. New
`DetectionMetrics.no_separation_evidence` field records when this fired.
Added
`tests/test_integrity.py::test_no_separation_evidence_accepts_everyone_on_identical_updates`
and `::test_weak_silhouette_accepts_everyone_even_with_real_outlier_absent`.

## E — Overloaded `TrustDecision.accepted` semantics (Section 18)

**Files:** `src/tfacd/runtime/contracts.py`,
`src/tfacd/trust_boundary/capability_enforcement.py`,
`src/tfacd/trust_boundary/boundary.py`

`accepted` was being used both for "trust boundary approved this plan" and,
by downstream readers, implicitly for "the action actually ran" — but
`accepted=True, executed_actions=[]` is a real, valid combination (e.g. an
approved plan whose executor failed).

**Fix:** added `TrustDecision.execution_status`
(`not_attempted`/`simulated`/`executed`/`failed`/`partially_executed`),
computed from a new `capability_enforcement.eligible_actions()` helper
(same whitelist/context/autonomy filtering as `enforce()`, but without
invoking the executor) compared against what `enforce()` actually executed.
`enforce()`'s own signature and behavior are unchanged, so no existing
caller/test needed to change for that function itself. Checked
`analytics/kpi.py` and `analytics/reputation.py` — both already treat
`accepted` correctly as a trust-approval/policy-violation signal, not an
execution-success signal, so they didn't need changes. Added four tests to
`tests/test_boundary.py` covering `simulated`, `not_attempted`, `failed`, and
`partially_executed`, plus one in `test_capability_enforcement.py` verifying
`eligible_actions()` matches `enforce()`'s filtering exactly.

## Sections 19 & 20 — Context-consistency scoring bugs

**File:** `src/tfacd/trust_boundary/context_consistency.py`

Two independent bugs in the same scoring function:
- **§19:** `action.target is None` counted as a full target match — so a
  missing target on a high-risk capability (`block_source`, `rate_limit`,
  `isolate_segment`) got the same credit as a correct one.
- **§20:** the score blended in a "confidence_alignment" term rewarding the
  agent's confidence for being close to the detector's — directly
  contradicting the LLM prompt's explicit instruction to report its own
  calibrated confidence, and gameable by simply copying the detector.

**Fix:** target is now mandatory (no credit for `None`) for the three
high-risk capabilities named above; the confidence-gap term was removed
entirely rather than patched, per the review's explicit recommendation.
Context-consistency is now purely "did the plan target the right entity."
Rewrote `tests/test_context_consistency.py` to match (the old
`test_mismatched_confidence_scores_lower` test encoded the flawed behavior
and was replaced with `test_confidence_no_longer_affects_context_consistency_score`),
and added `test_missing_target_on_high_risk_capability_scores_zero` /
`test_missing_target_on_capability_without_target_mapping_still_scores_full`.

## Section 21 — ASTB target validation & quota semantics

**File:** `src/tfacd/trust_boundary/preprocessing.py`

Two bugs:
- `action.target` received none of the length/canonicalization/obfuscation
  checks `action.parameters` values got, despite being one of the most
  security-sensitive fields (it's what network capabilities act on).
- `entity_action_quota_per_hour` was counting **decision events**, not
  **actions** — a 1-action plan and a 5-action plan both counted as "1"
  against the quota.

**Fix:** `target` now goes through the same length/canonicalize/obfuscation
pipeline as `parameters`. Quota now sums `len(capabilities)` from each past
`trust_decision` event's payload (boundary.py already records one entry per
action there) plus the current plan's action count, compared against the
quota. Added `test_hidden_base64_target_rejected`,
`test_oversized_target_truncated_and_rejected`, and
`test_hourly_quota_counts_actions_not_decisions` to
`tests/test_preprocessing.py`; updated `test_hourly_quota_exceeded_rejected`
to match the corrected counting (it previously logged events with no
`capabilities` payload key, which no longer registers as any actions at
all under the corrected logic).

**Not done:** the review also mentions "capability-specific parameter
schemas" as a further hardening step in the same section. That's a
larger, open-ended feature (defining a schema per capability) rather than a
single fixable bug, and wasn't specified concretely enough to implement
without risking scope creep — left for a follow-up conversation if wanted.

## G — Certification signs the wrong artifact (Section 14)

**Files:** `src/tfacd/integrity/certification.py`,
`src/tfacd/integrity/signing.py`, `scripts/certify_model.py`

The Ed25519 signature covered the **model file's raw bytes**. The
certified/uncertified `status` lives only in the manifest JSON alongside it,
which the signature said nothing about — an attacker able to edit the
manifest (but not re-sign anything) could flip
`metadata.status: "trained-uncertified"` to `"certified"` and the model
signature would still verify.

**Fix:** added `canonical_manifest_bytes()` (deterministic, sorted-key JSON
encoding of the manifest payload) and `sign_bytes()`/`verify_bytes()` in
`signing.py`. `certify_model.py --sign` and `certification.verify_release()`
now sign/verify the **canonical manifest** (which already contains the
model's sha256 + metadata + status together), not the model file directly.
The existing independent `sha256_ok` check still catches a model file being
swapped out from under an unchanged, correctly-signed manifest. `sign_file`/
`verify_file` remain available (as thin wrappers over the new byte-level
primitives) for signing arbitrary files — nothing else in the codebase used
them besides the certification path, which has been migrated.

Rewrote `tests/test_certification.py`'s `_make_release` helper and
`test_tampered_signature_fails` to sign/verify manifests instead of model
files, and added
`test_manifest_status_tampering_without_resigning_is_caught` — the exact
attack scenario the review described, now caught.

**Documented, not fixed (needs your input, not a code fix):** the review
also notes the public key is read from a local `artifacts/keys` path rather
than a deployment-pinned trust root. Where the trusted public key should
actually live and how it gets provisioned to verifying hosts is a deployment
decision that depends on how this project is actually deployed — inventing
a mechanism for that without knowing your deployment plan would just be
guessing, so it's called out explicitly in `certification.py`'s module
docstring and in the updated `SECURITY.md` as a known limitation instead.

## H — Cross-module consistency

**`configs/attack_scenarios.yaml` (Section 37):** the `ddos_segment_isolation`
scenario declared `expected_capability: isolate_segment`, but
`threat_context.yaml`'s `DDoS_TCP` entry (what the scenario's injected
traffic actually classifies as) only allows
`[rate_limit, block_source, start_capture, notify_soc]` for every `DDoS_*`
category — `isolate_segment` could structurally never be proposed for this
scenario. Fixed by changing `expected_capability` to `rate_limit` (already
policy-allowed for every DDoS category, and semantically closer to
"throttling a flood" than `block_source`), rather than widening
`DDoS_TCP`'s `allowed_playbooks` — that would have been a real policy change
affecting every DDoS category, not just this one demo scenario, and wasn't
asked for. Verified no test loads this real config file directly, so
nothing else needed updating.

**Executor factory duplication (Section 39):**
`capability_enforcement.build_executor_from_config()` and
`executor_factory.build_executor()` are two independent ways to build a
`CapabilityExecutor`, and only the latter is on any actual runtime path.
`build_executor_from_config` supports two drivers (`command`, `webhook`)
the canonical factory doesn't, so it isn't a strict duplicate to delete
outright (and `tests/test_executors.py` exercises it directly) — per the
review's own "cleanup, not a rewrite" framing, it's now clearly marked
legacy in its docstring, pointing at `executor_factory.build_executor()` as
the one to use for anything new.

**`run_submission_workflow.py` executor mismatch (Section 40):** this script
constructed `AdaptiveSemanticTrustBoundary` without passing `executor=` at
all, silently defaulting to `SimulatedExecutor` regardless of what
`configs/*.yaml`'s `trust_boundary.executor.mode` actually said — so this
script's behavior could diverge from the configured runtime with no error.
Fixed to build its executor via `executor_factory.build_executor(config)`,
the same canonical path `run_streaming_demo.py`/`run_attack_scenario.py`
already use.

**`SECURITY.md` (Section 42):** rewrote the "Security Design Notes" table,
which described an older/aspirational architecture:
- "SHA-256 hash pinned in `configs/*.yaml`" → replaced with the actual
  manifest+signature mechanism.
- "Differential-privacy noise via `dp_noise_scale`" → this option doesn't
  exist anywhere in the codebase; removed rather than left describing a
  feature that isn't there.
- "`executors.py` allow-list" → `executors.py` holds execution *drivers*,
  not the whitelist; the actual enforcement is
  `capability_enforcement.py` against `trust_policy.yaml` +
  `threat_context.yaml`.
Added a short "Known limitations" paragraph (quorum/detector limits, the
certification trust-root gap, no per-client update signing) so the doc
doesn't overstate what's implemented.

## Bonus fixes (P1, concrete, in scope — not in the A–H list but directly
called out by the review with a clear minimal fix)

**Section 26 — output sanitization leak path.**
`src/tfacd/trust_boundary/output_protection.py`'s `sanitize_decision()`
redacted only `TrustDecision.rationale`. `preprocessing.py`'s obfuscation
detector can embed a decoded preview directly into a `StageResult.reasons`
entry (e.g. "looks base64-encoded (decodes to: ...)"), which could carry
sensitive decoded content straight into the audit log, unredacted. Fixed to
redact every `StageResult.reasons` entry too. Added
`test_sanitize_decision_redacts_stage_result_reasons` to
`tests/test_output_protection.py`.

**Section 27 — audit log trusts on startup.**
`src/tfacd/trust_boundary/audit.py`'s `AuditLogger.__init__` read only the
last line's stored hash and continued appending from it, without ever
checking the *on-disk* chain was internally consistent — a log tampered with
before the process started would be silently accepted as ground truth.
Fixed: `verify_chain()` now runs once at construction; if it fails, a new
`AuditChainTamperedError` is raised by `append()` rather than the process
silently continuing a broken chain. Added
`test_new_logger_refuses_to_append_onto_tampered_existing_log` to
`tests/test_audit.py`.

## Explicitly not touched (Phase-II per the review's own framing, or genuinely open-ended)

RAG wiring/vocabulary alignment, analytics module refinements (drift,
forecasting, reputation, explainability), memory/tool-poisoning detection,
SBERT/TF-IDF scorer provenance recording, real per-client update
signing, genuine network-segment isolation (vs. the current
address-blocking implementation), and the streaming L=16 buffering (Choice
B) — all either explicitly deferred by the review itself or too
open-ended/deployment-dependent to implement without more direction from
you.

## A note on testing

Per your instructions, none of this was executed or dry-run in this
environment. Every file was verified with `python -m py_compile` (all
pass) and the two edited YAML configs were verified to parse. Test files
were updated/added alongside each behavioral change so the fixes are
demonstrated in the test suite, but you'll want to actually run
`pytest` in your own environment before relying on any of this.
