from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class ValidationResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    update_norm: float = 0.0
    reference_norm: float = 0.0


def validate_update(
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    max_abs_parameter: float,
    max_update_norm_ratio: float,
) -> ValidationResult:
    reasons: list[str] = []
    if candidate.keys() != reference.keys():
        reasons.append("state-dict keys mismatch")
        return ValidationResult(False, reasons)

    update_sq = 0.0
    ref_sq = 0.0
    for key in reference:
        c = np.asarray(candidate[key])
        r = np.asarray(reference[key])
        if c.shape != r.shape:
            reasons.append(f"shape mismatch for {key}: {c.shape} != {r.shape}")
            continue
        if not np.isfinite(c).all():
            reasons.append(f"non-finite values in {key}")
        if np.max(np.abs(c), initial=0.0) > max_abs_parameter:
            reasons.append(f"absolute parameter bound exceeded in {key}")
        delta = c.astype(np.float64) - r.astype(np.float64)
        update_sq += float(np.sum(delta * delta))
        ref_sq += float(np.sum(r.astype(np.float64) ** 2))

    update_norm = update_sq**0.5
    ref_norm = ref_sq**0.5
    ratio = update_norm / max(ref_norm, 1e-12)
    if ratio > max_update_norm_ratio:
        reasons.append(f"update norm ratio {ratio:.4f} exceeds {max_update_norm_ratio}")
    return ValidationResult(not reasons, reasons, update_norm, ref_norm)
