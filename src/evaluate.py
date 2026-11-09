"""Run the first deterministic *harness smoke test* (no model API calls).

Usage: python3 -m src.evaluate
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .validator import DECISIONS, decide

ROOT = Path(__file__).resolve().parent.parent


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if not cases:
        raise ValueError("No test cases found")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case_id detected")
    for case in cases:
        if case["expected"] not in DECISIONS:
            raise ValueError(f"Unknown expected label: {case['case_id']}")
    return cases


def run(dataset: Path, output: Path) -> dict:
    cases = load_cases(dataset)
    rows = []
    for case in cases:
        predicted, reason = decide(case)
        rows.append({
            "case_id": case["case_id"],
            "expected": case["expected"],
            "predicted": predicted,
            "correct": predicted == case["expected"],
            "reason": reason,
        })

    count = len(rows)
    correct = sum(row["correct"] for row in rows)
    unsafe = sum(row["predicted"] == "PASS" and row["expected"] != "PASS" for row in rows)
    nonpass_count = sum(row["expected"] != "PASS" for row in rows)
    summary = {
        "mode": "deterministic_harness_smoke_test_NOT_LLM_experiment",
        "cases": count,
        "label_counts": dict(Counter(row["expected"] for row in rows)),
        "accuracy": round(correct / count, 4),
        "false_approval_rate": round(unsafe / nonpass_count, 4) if nonpass_count else None,
        "note": "This validates dataset and reference rules only. Model comparison is not performed yet.",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "expected", "predicted", "correct", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "cases.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.dataset, args.out)
