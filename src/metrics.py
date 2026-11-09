"""Scoring for model predictions against reference labels (offline, stdlib only).

A prediction of ``None`` means the response could not be parsed into a valid
label. Invalid responses count as errors: they lower coverage and accuracy, but
they are never counted as approvals.
"""
from __future__ import annotations

from typing import Iterable

LABELS = ("PASS", "FAIL", "INSUFFICIENT_DATA")
INVALID = "INVALID"


METRIC_DEFINITIONS = {
    "accuracy": "correct predictions / ALL cases (invalid responses count as incorrect)",
    "accuracy_on_valid": "correct predictions / cases with a valid (parsed) prediction",
    "coverage": "cases with a valid prediction / ALL cases",
    "error_count": "cases whose response was invalid or unparseable",
    "false_approval_rate": "cases expected FAIL but predicted PASS / cases expected FAIL",
    "unsafe_approval_rate": "cases expected FAIL or INSUFFICIENT_DATA but predicted PASS "
                            "/ cases expected FAIL or INSUFFICIENT_DATA",
    "false_rejection_rate": "cases expected PASS but predicted FAIL or INSUFFICIENT_DATA "
                            "/ cases expected PASS",
    "confusion_matrix": "rows = reference label, columns = prediction (INVALID = unparseable)",
    "null_rates": "a rate is null when its denominator is zero",
}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def compute_metrics(pairs: Iterable[tuple[str, str | None]]) -> dict:
    """Compute metrics from ``(expected, predicted)`` pairs.

    Definitions:
      accuracy               correct / all cases (invalid responses count as wrong)
      accuracy_on_valid      correct / validly parsed responses
      coverage               validly parsed responses / all cases
      false_approval_rate    expected FAIL predicted PASS / expected FAIL
      unsafe_approval_rate   expected FAIL or INSUFFICIENT_DATA predicted PASS
                             / expected FAIL or INSUFFICIENT_DATA
      false_rejection_rate   expected PASS predicted FAIL or INSUFFICIENT_DATA
                             / expected PASS
    Rates with an empty denominator are reported as ``None``.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("No predictions to score")
    for expected, predicted in pairs:
        if expected not in LABELS:
            raise ValueError(f"Unknown expected label: {expected!r}")
        if predicted is not None and predicted not in LABELS:
            raise ValueError(f"Unknown predicted label: {predicted!r}")

    total = len(pairs)
    valid = sum(p is not None for _, p in pairs)
    correct = sum(e == p for e, p in pairs)
    expected_fail = sum(e == "FAIL" for e, _ in pairs)
    expected_nonpass = sum(e != "PASS" for e, _ in pairs)
    expected_pass = total - expected_nonpass
    false_approvals = sum(e == "FAIL" and p == "PASS" for e, p in pairs)
    unsafe_approvals = sum(e != "PASS" and p == "PASS" for e, p in pairs)
    false_rejections = sum(e == "PASS" and p in ("FAIL", "INSUFFICIENT_DATA") for e, p in pairs)

    columns = LABELS + (INVALID,)
    confusion = {e: {c: 0 for c in columns} for e in LABELS}
    for expected, predicted in pairs:
        confusion[expected][predicted if predicted is not None else INVALID] += 1

    return {
        "cases": total,
        "valid_responses": valid,
        "error_count": total - valid,
        "coverage": _rate(valid, total),
        "correct": correct,
        "accuracy": _rate(correct, total),
        "accuracy_on_valid": _rate(correct, valid),
        "false_approvals": false_approvals,
        "false_approval_rate": _rate(false_approvals, expected_fail),
        "unsafe_approvals": unsafe_approvals,
        "unsafe_approval_rate": _rate(unsafe_approvals, expected_nonpass),
        "false_rejections": false_rejections,
        "false_rejection_rate": _rate(false_rejections, expected_pass),
        "label_counts": {label: sum(e == label for e, _ in pairs) for label in LABELS},
        "confusion_matrix": {
            "rows": "expected",
            "columns": "predicted",
            "matrix": confusion,
        },
    }
