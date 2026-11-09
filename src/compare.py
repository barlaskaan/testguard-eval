"""Baseline vs guarded comparison on the SAME simulated model responses (offline).

Pipeline:
  1. Query the mock model ONCE per case (baseline arm).
  2. Feed those exact baseline predictions through the policy gate (guarded arm).
     The gate receives only the measurement fields of each case.
  3. Join reference labels only at scoring time and score both arms identically.

All outputs are SIMULATED and are NOT real LLM performance.

Usage:
  python3 -m src.compare --mock-profile injection_susceptible
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .evaluate import ROOT, load_cases
from .llm_baseline import (MOCK_PROFILES, PROMPT_VERSION, SIMULATED_TAG, MockClient,
                           ModelClient, parse_case_ids, predict, select_cases)
from .metrics import INVALID, METRIC_DEFINITIONS, compute_metrics
from .policy_gate import GATE_FIELDS, apply_gate

CSV_NAME = "compare.csv"
SUMMARY_NAME = "comparison_summary.json"


def gate_inputs(case: dict[str, Any]) -> dict[str, Any]:
    """Project a case onto measurement fields only (no note, no label)."""
    return {field: case.get(field) for field in GATE_FIELDS}


def apply_gate_to_predictions(predictions: list[dict], cases: list[dict]) -> list[dict]:
    """Guarded arm: reuse the given baseline predictions; no new model calls."""
    measurements = {case["case_id"]: gate_inputs(case) for case in cases}
    rows = []
    for pred in predictions:
        decision = apply_gate(pred["predicted"], measurements[pred["case_id"]])
        rows.append({
            "case_id": pred["case_id"],
            "baseline_prediction": pred["predicted"],
            "guarded_prediction": decision.guarded,
            "vetoed": decision.vetoed,
            "gate_explanation": decision.explanation,
            "baseline_parse_error": pred["parse_error"],
            "raw_response": pred["raw_response"],
        })
    return rows


def score_arms(rows: list[dict], cases: list[dict]) -> tuple[list[dict], dict, dict]:
    """Join reference labels (scoring time only) and score both arms."""
    expected_by_id = {case["case_id"]: case["expected"] for case in cases}
    scored = []
    for row in rows:
        expected = expected_by_id[row["case_id"]]
        scored.append({
            **row,
            "expected": expected,
            "baseline_correct": row["baseline_prediction"] == expected,
            "guarded_correct": row["guarded_prediction"] == expected,
        })
    baseline = compute_metrics((r["expected"], r["baseline_prediction"]) for r in scored)
    guarded = compute_metrics((r["expected"], r["guarded_prediction"]) for r in scored)
    return scored, baseline, guarded


def _label(value: str | None) -> str:
    return value if value is not None else INVALID


def run_comparison(cases: list[dict], client: ModelClient) -> tuple[list[dict], dict]:
    predictions = predict(cases, client)  # exactly one model call per case
    gated = apply_gate_to_predictions(predictions, cases)
    rows, baseline, guarded = score_arms(gated, cases)
    changed = [
        {
            "case_id": r["case_id"],
            "expected": r["expected"],
            "baseline_prediction": _label(r["baseline_prediction"]),
            "guarded_prediction": _label(r["guarded_prediction"]),
            "explanation": r["gate_explanation"],
        }
        for r in rows if r["vetoed"]
    ]
    summary = {
        "result_type": SIMULATED_TAG,
        "simulated": True,
        "warning": (
            "SIMULATED: baseline responses come from an offline mock, not a language model. "
            "The gate encodes the same acceptance policy used to write the reference labels, "
            "so improvements are expected by construction and are NOT evidence of real LLM safety."
        ),
        "model": client.name,
        "prompt_version": PROMPT_VERSION,
        "cases": len(rows),
        "case_ids": [r["case_id"] for r in rows],
        "model_calls": len(predictions),
        "same_baseline_predictions_used_for_both_arms": True,
        "gate_fields_read": list(GATE_FIELDS),
        "vetoed_count": len(changed),
        "changed_cases": changed,
        "metric_definitions": METRIC_DEFINITIONS,
        "baseline": baseline,
        "guarded": guarded,
    }
    return rows, summary


def write_outputs(rows: list[dict], summary: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / CSV_NAME, out_dir / SUMMARY_NAME
    fields = ["result_type", "model", "prompt_version", "case_id", "expected",
              "baseline_prediction", "guarded_prediction", "vetoed", "baseline_correct",
              "guarded_correct", "baseline_parse_error", "gate_explanation", "raw_response"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "result_type": SIMULATED_TAG,
                "model": summary["model"],
                "prompt_version": PROMPT_VERSION,
                "case_id": r["case_id"],
                "expected": r["expected"],
                "baseline_prediction": _label(r["baseline_prediction"]),
                "guarded_prediction": _label(r["guarded_prediction"]),
                "vetoed": r["vetoed"],
                "baseline_correct": r["baseline_correct"],
                "guarded_correct": r["guarded_correct"],
                "baseline_parse_error": r["baseline_parse_error"] or "",
                "gate_explanation": r["gate_explanation"],
                "raw_response": r["raw_response"],
            })
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def _fmt(value: Any) -> str:
    return "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def format_report(summary: dict) -> str:
    b, g = summary["baseline"], summary["guarded"]
    lines = [
        f"*** {SIMULATED_TAG} ***",
        summary["warning"],
        "",
        f"Model: {summary['model']}   Cases: {summary['cases']}   Model calls: {summary['model_calls']}",
        "",
        f"{'metric':<24}{'baseline':>10}{'guarded':>10}   denominator",
    ]
    rows = [
        ("coverage", "all cases"),
        ("accuracy", "all cases"),
        ("accuracy_on_valid", "valid predictions"),
        ("false_approval_rate", "expected FAIL"),
        ("unsafe_approval_rate", "expected FAIL or INSUFFICIENT_DATA"),
        ("false_rejection_rate", "expected PASS"),
        ("error_count", "count"),
        ("false_approvals", "count"),
        ("unsafe_approvals", "count"),
        ("false_rejections", "count"),
    ]
    for key, denom in rows:
        lines.append(f"{key:<24}{_fmt(b[key]):>10}{_fmt(g[key]):>10}   {denom}")
    for arm, metrics in (("baseline", b), ("guarded", g)):
        matrix = metrics["confusion_matrix"]["matrix"]
        cols = list(next(iter(matrix.values())))
        lines += ["", f"Confusion matrix ({arm}; rows=expected, cols=predicted)",
                  f"{'':<20}" + "".join(f"{c:>19}" for c in cols)]
        for exp, counts in matrix.items():
            lines.append(f"{exp:<20}" + "".join(f"{counts[c]:>19}" for c in cols))
    lines += ["", f"Decisions vetoed: {summary['vetoed_count']}"]
    for c in summary["changed_cases"]:
        lines.append(f"  {c['case_id']}: {c['baseline_prediction']} -> {c['guarded_prediction']} "
                     f"(expected {c['expected']}) | {c['explanation']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline baseline vs guarded comparison (SIMULATED).")
    parser.add_argument("--mock-profile", choices=sorted(MOCK_PROFILES), default="injection_susceptible")
    parser.add_argument("--case-ids", help="comma-separated case IDs")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "cases.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    try:
        cases = select_cases(load_cases(args.dataset), parse_case_ids(args.case_ids))
    except ValueError as exc:
        parser.error(str(exc))

    rows, summary = run_comparison(cases, MockClient(args.mock_profile))
    csv_path, json_path = write_outputs(rows, summary, args.out)
    print(format_report(summary))
    print(f"\nWrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
