"""Manual (copy/paste) real-LLM evaluation through a chat interface. No API calls.

This module never queries a model and never generates model responses. A human
pastes each exported prompt into a NEW chat conversation and saves the model's
raw answer, unedited, to manual_responses/<CASE_ID>.txt.

Subcommands:
  export  Write paste-ready prompts (policy + measurements only; never labels)
          to manual_prompts/, plus a manifest and a metadata template.
  score   Read the saved raw answers, parse them with the strict Stage 2 parser,
          apply the policy gate to the SAME answers, and score both arms.

Usage:
  python3 -m src.manual_eval export --case-ids TC001,TC004,TC011
  python3 -m src.manual_eval score  --case-ids TC001,TC004,TC011
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .compare import apply_gate_to_predictions, score_arms
from .evaluate import ROOT, load_cases
from .llm_baseline import PROMPT_VERSION, SIMULATED_TAG, build_prompt, parse_case_ids, parse_response, select_cases
from .metrics import INVALID, METRIC_DEFINITIONS
from .normalization import (FORMAT_NORMALIZED, NORMALIZATION_RULES, NORMALIZATION_VERSION, STRICT,
                            format_normalized_parse)

REAL_TAG = "REAL_MANUAL_LLM_EVALUATION"
assert REAL_TAG != SIMULATED_TAG

PROMPTS_DIR = ROOT / "manual_prompts"
RESPONSES_DIR = ROOT / "manual_responses"
MANIFEST_NAME = "manifest.json"
METADATA_NAME = "metadata.json"
CSV_NAME = "manual_comparison.csv"
SUMMARY_NAME = "manual_summary.json"

PLACEHOLDER = "FILL_IN"
REQUIRED_METADATA = ("model_name", "experiment_date", "interface", "chat_settings")
METADATA_TEMPLATE = {
    "model_name": f"{PLACEHOLDER}: exact model name shown in the chat model picker",
    "experiment_date": f"{PLACEHOLDER}: YYYY-MM-DD",
    "interface": f"{PLACEHOLDER}: e.g. claude.ai web chat, Pro plan, manual copy/paste",
    "chat_settings": f"{PLACEHOLDER}: extended thinking on/off, web search, styles, memory, "
                     "custom/project instructions, attachments",
    "operator_limitations": f"{PLACEHOLDER}: anything that deviated from the protocol, or empty string",
}

METHODOLOGY_LIMITATIONS = [
    "Manual chat UI: prompts and answers were copied by hand; transcription errors are possible.",
    "Unknown sampling parameters: temperature, top_p, max tokens and any server-side system "
    "prompt of the chat product are not controlled or recorded.",
    "No true system-role control: the policy text is pasted as a user message, not sent as an API "
    "system prompt, so it competes with the product's own system prompt.",
    "Product features (memory, custom/project instructions, styles, extended thinking, tools) may "
    "influence answers; see chat_settings.",
    "Single sample per case; no repeated runs, so variance is unmeasured.",
    "Very small sample size; rates move in large steps and support no statistical inference.",
    "The policy gate encodes the same policy as the reference labels, so guarded-arm improvements "
    "are expected by construction.",
    "Synthetic cases only; not a safety assessment of real equipment.",
]


# --------------------------------------------------------------------------- export

def build_manual_prompt(case: dict[str, Any]) -> str:
    """One paste-ready message: policy instructions followed by the case data.

    Reuses the Stage 2 prompt verbatim. A chat UI has no separate system role,
    so the system text is placed first in the same user message.
    """
    prompt = build_prompt(case)
    return f"{prompt.system}\n\n{prompt.user}\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_prompts(cases: list[dict], prompts_dir: Path, responses_dir: Path) -> dict:
    prompts_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for case in cases:
        text = build_manual_prompt(case)
        path = prompts_dir / f"{case['case_id']}.txt"
        path.write_text(text, encoding="utf-8")
        entries[case["case_id"]] = {"prompt_file": path.name, "prompt_sha256": sha256_text(text)}
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Prompt files contain the decision policy and case measurements only.",
        "cases": entries,
    }
    (prompts_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metadata_path = responses_dir / METADATA_NAME
    if not metadata_path.exists():  # never overwrite user-entered metadata
        metadata_path.write_text(json.dumps(METADATA_TEMPLATE, indent=2) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- score

class MetadataError(ValueError):
    pass


def load_metadata(path: Path) -> dict:
    if not path.exists():
        raise MetadataError(f"Metadata file not found: {path}. Run export first, then fill it in.")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataError(f"Metadata is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise MetadataError("Metadata must be a JSON object")
    problems = []
    for key in REQUIRED_METADATA:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{key} is required")
    for key, value in metadata.items():
        if isinstance(value, str) and value.strip().startswith(PLACEHOLDER):
            problems.append(f"{key} still contains the {PLACEHOLDER} placeholder")
    if not problems:
        try:
            date.fromisoformat(metadata["experiment_date"])
        except ValueError:
            problems.append("experiment_date must be YYYY-MM-DD")
    if problems:
        raise MetadataError("Invalid metadata: " + "; ".join(problems))
    return metadata


def read_raw_response(path: Path) -> tuple[str | None, str | None, str | None]:
    """Return (raw_text, sha256_of_bytes, read_error). The text is never altered."""
    if not path.exists():
        return None, None, "missing_response_file"
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        return data.decode("utf-8"), digest, None
    except UnicodeDecodeError:
        return None, digest, "undecodable_response"


def collect_predictions(cases: list[dict], responses_dir: Path) -> tuple[list[dict], dict]:
    """Parse saved answers. Missing/unreadable/malformed answers stay errors (None)."""
    predictions, files = [], {}
    for case in cases:
        cid = case["case_id"]
        path = responses_dir / f"{cid}.txt"
        raw, digest, read_error = read_raw_response(path)
        if read_error:
            predicted, error = None, read_error
        else:
            parsed = parse_response(raw)
            predicted, error = parsed.decision, parsed.error
        predictions.append({"case_id": cid, "predicted": predicted, "parse_error": error,
                            "raw_response": raw if raw is not None else ""})
        files[cid] = {"response_file": path.name, "response_sha256": digest, "read_error": read_error}
    return predictions, files


def prompt_integrity(cases: list[dict], prompts_dir: Path) -> dict:
    """Check the prompts regenerate exactly as exported (detects prompt drift)."""
    manifest_path = prompts_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    out = {}
    for case in cases:
        current = sha256_text(build_manual_prompt(case))
        exported = (manifest or {}).get("cases", {}).get(case["case_id"], {}).get("prompt_sha256")
        out[case["case_id"]] = {
            "prompt_sha256": current,
            "prompt_matches_export": None if exported is None else exported == current,
        }
    return out


def _label(value: str | None) -> str:
    return value if value is not None else INVALID


def normalize_predictions(predictions: list[dict], files: dict) -> tuple[list[dict], dict]:
    """FORMAT_NORMALIZED predictions from the same raw answers (in-memory only).

    Read errors (missing/undecodable files) stay errors; nothing is guessed.
    """
    normalized, records = [], {}
    for pred in predictions:
        cid = pred["case_id"]
        read_error = files[cid]["read_error"]
        if read_error:
            decision, error, applied, transformations, digest = None, read_error, False, (), None
        else:
            result = format_normalized_parse(pred["raw_response"])
            decision, error = result.decision, result.error
            applied, transformations, digest = result.applied, result.transformations, result.normalized_sha256
        normalized.append({**pred, "predicted": decision, "parse_error": error})
        records[cid] = {
            "normalization_applied": applied,
            "normalization_transformations": list(transformations),
            "normalized_sha256": digest,
            "original_response_sha256": files[cid]["response_sha256"],
        }
    return normalized, records


def _changed(rows: list[dict]) -> list[dict]:
    return [
        {"case_id": r["case_id"], "expected": r["expected"],
         "baseline_prediction": _label(r["baseline_prediction"]),
         "guarded_prediction": _label(r["guarded_prediction"]),
         "explanation": r["gate_explanation"]}
        for r in rows if r["vetoed"]
    ]


def run_score(cases: list[dict], responses_dir: Path, prompts_dir: Path, metadata: dict) -> tuple[list[dict], dict]:
    predictions, files = collect_predictions(cases, responses_dir)
    # PRIMARY (STRICT): the SAME parsed real answers feed both arms; labels are joined only in score_arms.
    gated = apply_gate_to_predictions(predictions, cases)
    rows, baseline, guarded = score_arms(gated, cases)
    # SECONDARY (FORMAT_NORMALIZED): same raw answers, minimal in-memory quote normalization.
    norm_predictions, norm_records = normalize_predictions(predictions, files)
    norm_rows, norm_baseline, norm_guarded = score_arms(apply_gate_to_predictions(norm_predictions, cases), cases)
    norm_by_id = {r["case_id"]: r for r in norm_rows}
    integrity = prompt_integrity(cases, prompts_dir)
    for row in rows:
        cid = row["case_id"]
        row.update(files[cid])
        row.update(integrity[cid])
        n = norm_by_id[cid]
        row.update(norm_records[cid])
        row.update({
            "normalized_prediction": n["baseline_prediction"],
            "normalized_parse_error": n["baseline_parse_error"],
            "normalized_guarded_prediction": n["guarded_prediction"],
            "normalized_vetoed": n["vetoed"],
            "normalized_correct": n["baseline_correct"],
            "normalized_guarded_correct": n["guarded_correct"],
        })

    def _mode_row(b, g):
        return {k: {"baseline": b[k], "guarded": g[k]}
                for k in ("coverage", "accuracy", "accuracy_on_valid", "error_count",
                          "false_approval_rate", "unsafe_approval_rate", "false_rejection_rate")}

    summary = {
        "result_type": REAL_TAG,
        "simulated": False,
        "source": "Raw answers pasted by a human from a chat interface; no API calls, no mock responses.",
        "metadata": metadata,
        "prompt_version": PROMPT_VERSION,
        "cases": len(rows),
        "case_ids": [r["case_id"] for r in rows],
        "missing_responses": [r["case_id"] for r in rows if r["read_error"] == "missing_response_file"],
        "invalid_responses": [r["case_id"] for r in rows if r["baseline_prediction"] is None],
        "prompt_integrity_ok": all(r["prompt_matches_export"] is not False for r in rows),
        "same_baseline_predictions_used_for_both_arms": True,
        "vetoed_count": sum(r["vetoed"] for r in rows),
        "changed_cases": _changed(rows),
        "methodology_limitations": METHODOLOGY_LIMITATIONS,
        "metric_definitions": METRIC_DEFINITIONS,
        # Top-level "baseline"/"guarded" are the PRIMARY STRICT results (unchanged meaning).
        "primary_evaluation_mode": STRICT,
        "baseline": baseline,
        "guarded": guarded,
        "mode_comparison": {
            STRICT: _mode_row(baseline, guarded),
            FORMAT_NORMALIZED: _mode_row(norm_baseline, norm_guarded),
        },
        "format_normalized": {
            "evaluation_mode": FORMAT_NORMALIZED,
            "role": "SECONDARY analysis; report alongside, never instead of, STRICT results.",
            "normalization_version": NORMALIZATION_VERSION,
            "rules": NORMALIZATION_RULES,
            "original_files_modified": False,
            "normalized_case_ids": [r["case_id"] for r in rows if r["normalization_applied"]],
            "still_invalid_case_ids": [r["case_id"] for r in rows if r["normalized_prediction"] is None],
            "per_case": {
                r["case_id"]: {
                    "strict_prediction": _label(r["baseline_prediction"]),
                    "strict_parse_error": r["baseline_parse_error"],
                    "normalized_prediction": _label(r["normalized_prediction"]),
                    "normalized_parse_error": r["normalized_parse_error"],
                    "normalization_applied": r["normalization_applied"],
                    "transformations": r["normalization_transformations"],
                    "original_response_sha256": r["original_response_sha256"],
                    "normalized_sha256": r["normalized_sha256"],
                }
                for r in rows
            },
            "vetoed_count": sum(n["vetoed"] for n in norm_rows),
            "changed_cases": _changed(norm_rows),
            "baseline": norm_baseline,
            "guarded": norm_guarded,
        },
    }
    return rows, summary


def write_outputs(rows: list[dict], summary: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / CSV_NAME, out_dir / SUMMARY_NAME
    meta = summary["metadata"]
    fields = ["result_type", "model_name", "experiment_date", "interface", "prompt_version",
              "case_id", "expected", "baseline_prediction", "guarded_prediction", "vetoed",
              "baseline_correct", "guarded_correct", "baseline_parse_error", "gate_explanation",
              "response_file", "response_sha256", "prompt_sha256", "prompt_matches_export",
              "normalized_prediction", "normalized_guarded_prediction", "normalized_correct",
              "normalized_guarded_correct", "normalized_parse_error", "normalization_applied",
              "normalization_transformations", "normalized_sha256", "raw_response"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "result_type": REAL_TAG,
                "model_name": meta["model_name"],
                "experiment_date": meta["experiment_date"],
                "interface": meta["interface"],
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
                "response_file": r["response_file"],
                "response_sha256": r["response_sha256"] or "",
                "prompt_sha256": r["prompt_sha256"],
                "prompt_matches_export": "" if r["prompt_matches_export"] is None else r["prompt_matches_export"],
                "normalized_prediction": _label(r["normalized_prediction"]),
                "normalized_guarded_prediction": _label(r["normalized_guarded_prediction"]),
                "normalized_correct": r["normalized_correct"],
                "normalized_guarded_correct": r["normalized_guarded_correct"],
                "normalized_parse_error": r["normalized_parse_error"] or "",
                "normalization_applied": r["normalization_applied"],
                "normalization_transformations": "; ".join(
                    f"{t['from']}->{t['to']} x{t['count']}" for t in r["normalization_transformations"]),
                "normalized_sha256": r["normalized_sha256"] or "",
                "raw_response": r["raw_response"],
            })
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def _fmt(value: Any) -> str:
    return "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def format_report(summary: dict, rows: list[dict]) -> str:
    b, g, meta = summary["baseline"], summary["guarded"], summary["metadata"]
    lines = [
        f"*** {REAL_TAG} ***",
        f"Model: {meta['model_name']}   Date: {meta['experiment_date']}   Interface: {meta['interface']}",
        f"Cases: {summary['cases']}   Missing responses: {len(summary['missing_responses'])}   "
        f"Invalid responses: {len(summary['invalid_responses'])}",
        "",
        f"{'case':<8}{'expected':<20}{'baseline':<20}{'guarded':<20}parse_error",
    ]
    for r in rows:
        lines.append(f"{r['case_id']:<8}{r['expected']:<20}{_label(r['baseline_prediction']):<20}"
                     f"{_label(r['guarded_prediction']):<20}{r['baseline_parse_error'] or ''}")
    lines += ["", f"{'metric':<24}{'baseline':>10}{'guarded':>10}   denominator"]
    for key, denom in [("coverage", "all cases"), ("accuracy", "all cases"),
                       ("accuracy_on_valid", "valid predictions"),
                       ("false_approval_rate", "expected FAIL"),
                       ("unsafe_approval_rate", "expected FAIL or INSUFFICIENT_DATA"),
                       ("false_rejection_rate", "expected PASS"), ("error_count", "count")]:
        lines.append(f"{key:<24}{_fmt(b[key]):>10}{_fmt(g[key]):>10}   {denom}")
    lines += ["", f"Decisions vetoed: {summary['vetoed_count']}"]
    lines += [f"  {c['case_id']}: {c['baseline_prediction']} -> {c['guarded_prediction']} | {c['explanation']}"
              for c in summary["changed_cases"]]
    fn = summary["format_normalized"]
    lines += ["", f"--- SECONDARY: {FORMAT_NORMALIZED} ({fn['normalization_version']}; original files unchanged) ---",
              f"{'case':<8}{'strict':<20}{'normalized':<20}{'norm+gate':<20}transformations"]
    for r in rows:
        transforms = ", ".join(f"{t['from']}->{t['to']} x{t['count']}" for t in r["normalization_transformations"])
        lines.append(f"{r['case_id']:<8}{_label(r['baseline_prediction']):<20}{_label(r['normalized_prediction']):<20}"
                     f"{_label(r['normalized_guarded_prediction']):<20}{transforms or '-'}")
    lines += ["", f"{'metric (baseline arm)':<24}{'STRICT':>10}{'NORMALIZED':>12}"]
    for key in ("coverage", "accuracy", "accuracy_on_valid", "error_count",
                "false_approval_rate", "unsafe_approval_rate", "false_rejection_rate"):
        lines.append(f"{key:<24}{_fmt(b[key]):>10}{_fmt(fn['baseline'][key]):>12}")
    lines.append(f"Normalized-mode vetoes: {fn['vetoed_count']}")
    if not summary["prompt_integrity_ok"]:
        lines += ["", "WARNING: some prompts no longer match the exported manifest (prompt drift)."]
    lines += ["", "Limitations: manual UI, unknown sampling parameters, no API system role, "
              "single sample, tiny n. See manual_summary.json."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual chat-interface evaluation (no API calls).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "score"):
        p = sub.add_parser(name)
        p.add_argument("--case-ids", required=True, help="comma-separated case IDs, e.g. TC001,TC004,TC011")
        p.add_argument("--dataset", type=Path, default=ROOT / "data" / "cases.jsonl")
        p.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
        p.add_argument("--responses-dir", type=Path, default=RESPONSES_DIR)
        if name == "score":
            p.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    try:
        cases = select_cases(load_cases(args.dataset), parse_case_ids(args.case_ids))
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "export":
        manifest = export_prompts(cases, args.prompts_dir, args.responses_dir)
        for cid, entry in manifest["cases"].items():
            print(f"Wrote {args.prompts_dir / entry['prompt_file']}")
        print(f"Wrote {args.prompts_dir / MANIFEST_NAME}")
        print(f"Metadata template: {args.responses_dir / METADATA_NAME} (fill in before scoring)")
        print(f"Save each raw answer, unedited, to {args.responses_dir}/<CASE_ID>.txt")
        return 0

    try:
        metadata = load_metadata(args.responses_dir / METADATA_NAME)
    except MetadataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows, summary = run_score(cases, args.responses_dir, args.prompts_dir, metadata)
    csv_path, json_path = write_outputs(rows, summary, args.out)
    print(format_report(summary, rows))
    print(f"\nWrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
