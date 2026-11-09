"""Independent one-sided safety gate (veto) applied after a model decision.

The gate reads ONLY the raw measurement fields ``temperature_c``,
``vibration_g`` and ``calibrated``. It never reads ``technician_note``, never
reads reference labels, and never calls the reference validator.

It is one-sided: it can only downgrade a ``PASS``.
  * PASS + a definite violation              -> FAIL
  * PASS + missing/invalid evidence only     -> INSUFFICIENT_DATA
  * PASS with all evidence valid and in limits -> PASS (unchanged)
  * FAIL / INSUFFICIENT_DATA                 -> unchanged
  * invalid model response (None)            -> stays an error (None)

Measurements are never coerced: strings such as "38.0", booleans in numeric
fields, NaN and +/-infinity are invalid, and "calibrated" must be the JSON
boolean ``true`` (1, "true", "yes" are invalid).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

# Thresholds are shared with the written policy so they cannot drift apart;
# the checking logic below is an independent implementation.
from .validator import MAX_TEMPERATURE_C, MAX_VIBRATION_G

GATE_FIELDS = ("temperature_c", "vibration_g", "calibrated")
LABELS = ("PASS", "FAIL", "INSUFFICIENT_DATA")


@dataclass(frozen=True)
class Assessment:
    violations: tuple[str, ...]
    missing: tuple[str, ...]
    invalid: tuple[str, ...]

    @property
    def status(self) -> str:
        if self.violations:
            return "VIOLATION"  # a known violation takes priority over missing data
        if self.missing or self.invalid:
            return "INCOMPLETE"
        return "OK"

    def describe(self) -> str:
        parts = list(self.violations)
        parts += [f"{name} is missing" for name in self.missing]
        parts += list(self.invalid)
        return "; ".join(parts) if parts else "all measurements valid and within limits"


@dataclass(frozen=True)
class GateDecision:
    original: str | None
    guarded: str | None
    vetoed: bool
    explanation: str
    assessment: Assessment


def _check_limit(name, value, limit, unit, violations, missing, invalid) -> None:
    if value is None:
        missing.append(name)
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        invalid.append(f"{name} is not a number ({type(value).__name__} {value!r})")
    elif not math.isfinite(value):
        invalid.append(f"{name} is not finite ({value!r})")
    elif value > limit:
        violations.append(f"{name}={value} exceeds {limit} {unit}")


def assess(measurements: Mapping[str, Any]) -> Assessment:
    """Assess raw measurements. Only GATE_FIELDS are read, each via ``.get``."""
    temperature = measurements.get("temperature_c")
    vibration = measurements.get("vibration_g")
    calibrated = measurements.get("calibrated")

    violations: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    _check_limit("temperature_c", temperature, MAX_TEMPERATURE_C, "C", violations, missing, invalid)
    _check_limit("vibration_g", vibration, MAX_VIBRATION_G, "g", violations, missing, invalid)
    if calibrated is None:
        missing.append("calibrated")
    elif calibrated is False:
        violations.append("calibrated is false")
    elif calibrated is not True:
        invalid.append(f"calibrated is not a boolean ({type(calibrated).__name__} {calibrated!r})")
    return Assessment(tuple(violations), tuple(missing), tuple(invalid))


def apply_gate(baseline: str | None, measurements: Mapping[str, Any]) -> GateDecision:
    """Apply the one-sided veto to a single baseline decision."""
    if baseline is not None and baseline not in LABELS:
        raise ValueError(f"Unknown baseline decision: {baseline!r}")
    assessment = assess(measurements)

    if baseline is None:
        return GateDecision(None, None, False, "Invalid model response preserved as an error.", assessment)
    if baseline != "PASS":
        return GateDecision(baseline, baseline, False, f"{baseline} preserved (gate only vetoes PASS).", assessment)
    if assessment.status == "VIOLATION":
        return GateDecision("PASS", "FAIL", True, f"Vetoed PASS -> FAIL: {assessment.describe()}.", assessment)
    if assessment.status == "INCOMPLETE":
        return GateDecision("PASS", "INSUFFICIENT_DATA", True,
                            f"Vetoed PASS -> INSUFFICIENT_DATA: {assessment.describe()}.", assessment)
    return GateDecision("PASS", "PASS", False, "PASS preserved: all measurements valid and within limits.", assessment)
