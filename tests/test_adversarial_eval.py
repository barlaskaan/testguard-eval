"""Tests for the Stage 5 paired adversarial benchmark.

All "responses" here are SYNTHETIC FIXTURES written into temporary directories.
No model is queried, and the real adversarial_responses/ and manual_responses/
folders are never read or written.
"""
import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src import adversarial_eval as ae
from src import manual_eval as me
from src.llm_baseline import MODEL_INPUT_FIELDS, SYSTEM_PROMPT, build_prompt, model_inputs
from src.validator import decide

GOOD_METADATA = {
    "model_name": "synthetic-fixture-model",
    "experiment_date": "2026-11-09",
    "interface": "synthetic test fixture (no chat used)",
    "chat_settings": "n/a (unit test)",
    "operator_limitations": "",
}


def _json(decision):
    return json.dumps({"decision": decision, "reason": "synthetic fixture"})


def _raw_cases():
    with ae.DATASET.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Workspace:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.base = base
        self.prompts, self.responses, self.out = base / "prompts", base / "responses", base / "results"

    def close(self):
        self._tmp.cleanup()

    def run(self, command, dataset=None):
        argv = [command, "--prompts-dir", str(self.prompts), "--responses-dir", str(self.responses)]
        if dataset is not None:
            argv += ["--dataset", str(dataset)]
        if command == "score":
            argv += ["--out", str(self.out)]
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ae.main(argv)
        return code, out.getvalue(), err.getvalue()

    def write_metadata(self):
        self.responses.mkdir(parents=True, exist_ok=True)
        (self.responses / me.METADATA_NAME).write_text(json.dumps(GOOD_METADATA), encoding="utf-8")

    def write_response(self, cid, text):
        self.responses.mkdir(parents=True, exist_ok=True)
        (self.responses / f"{cid}.txt").write_bytes(text.encode("utf-8") if isinstance(text, str) else text)

    def write_dataset(self, cases):
        path = self.base / "dataset.jsonl"
        path.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
        return path

    def summary(self):
        return json.loads((self.out / ae.SUMMARY_NAME).read_text(encoding="utf-8"))

    def csv_rows(self):
        with (self.out / ae.CSV_NAME).open(encoding="utf-8", newline="") as handle:
            return {row["case_id"]: row for row in csv.DictReader(handle)}


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.close)
        self.cases = ae.load_adversarial_cases()
        self.by_id = {c["case_id"]: c for c in self.cases}


# --------------------------------------------------------------------------- dataset integrity

class DatasetIntegrityTests(BaseCase):
    def test_six_pairs_twelve_variants_with_unique_ids(self):
        self.assertEqual(len(self.cases), 12)
        self.assertEqual(len({c["case_id"] for c in self.cases}), 12)
        pairs = sorted({c["pair_id"] for c in self.cases})
        self.assertEqual(pairs, [f"ADV0{i}" for i in range(1, 7)])
        for c in self.cases:
            self.assertEqual(c["case_id"], f"{c['pair_id']}-{c['variant']}")

    def test_base_label_balance_two_each(self):
        labels = sorted(c["expected"] for c in self.cases if c["variant"] == ae.CLEAN)
        self.assertEqual(labels, ["FAIL", "FAIL", "INSUFFICIENT_DATA", "INSUFFICIENT_DATA", "PASS", "PASS"])

    def test_each_strategy_used_exactly_once(self):
        strategies = sorted(c["attack_strategy"] for c in self.cases if c["variant"] == ae.ATTACK)
        self.assertEqual(strategies, sorted(ae.STRATEGIES))
        for c in self.cases:
            if c["variant"] == ae.CLEAN:
                self.assertIsNone(c["attack_strategy"])
                self.assertIsNone(c["attack_target"])

    def test_pairs_share_measurements_and_label_but_not_note(self):
        for pair_id in {c["pair_id"] for c in self.cases}:
            clean, attack = self.by_id[f"{pair_id}-CLEAN"], self.by_id[f"{pair_id}-ATTACK"]
            with self.subTest(pair=pair_id):
                for field in ae.TRUSTED_FIELDS + ("expected",):
                    self.assertEqual(ae._canon(clean[field]), ae._canon(attack[field]))
                self.assertNotEqual(clean["technician_note"], attack["technician_note"])

    def test_labels_match_written_policy(self):
        for c in self.cases:
            with self.subTest(case=c["case_id"]):
                self.assertEqual(c["expected"], decide(c)[0])

    def test_attack_target_differs_from_expected_and_notes_never_state_expected_label(self):
        for c in self.cases:
            with self.subTest(case=c["case_id"]):
                if c["variant"] == ae.ATTACK:
                    self.assertNotEqual(c["attack_target"], c["expected"])
                self.assertNotIn(c["expected"], c["technician_note"])

    def test_clean_notes_contain_no_instructions_or_labels(self):
        for c in self.cases:
            if c["variant"] != ae.CLEAN:
                continue
            note = c["technician_note"].lower()
            for word in ("pass", "fail", "insufficient", "approve", "ignore", "system", "decision"):
                with self.subTest(case=c["case_id"], word=word):
                    self.assertNotIn(word, note)

    def _assert_rejected(self, cases, fragment):
        with self.assertRaises(ae.AdversarialDatasetError) as ctx:
            ae.validate_dataset(cases)
        self.assertIn(fragment, str(ctx.exception))

    def test_validator_rejects_measurement_mismatch(self):
        cases = copy.deepcopy(_raw_cases())
        cases[1]["vibration_g"] = 1.5
        self._assert_rejected(cases, "vibration_g differs")

    def test_validator_rejects_type_mismatch(self):
        cases = copy.deepcopy(_raw_cases())
        cases[1]["temperature_c"] = 36  # int vs float 36.0
        self._assert_rejected(cases, "temperature_c differs")

    def test_validator_rejects_label_mismatch_and_policy_disagreement(self):
        cases = copy.deepcopy(_raw_cases())
        cases[1]["expected"] = "FAIL"
        self._assert_rejected(cases, "expected differs")
        self._assert_rejected(cases, "written policy gives PASS")

    def test_validator_rejects_identical_notes(self):
        cases = copy.deepcopy(_raw_cases())
        cases[1]["technician_note"] = cases[0]["technician_note"]
        self._assert_rejected(cases, "notes are identical")

    def test_validator_rejects_missing_variant_and_duplicates(self):
        cases = copy.deepcopy(_raw_cases())
        self._assert_rejected(cases[1:], "exactly one CLEAN and one ATTACK")
        dup = copy.deepcopy(cases) + [copy.deepcopy(cases[0])]
        self._assert_rejected(dup, "duplicate CLEAN")

    def test_validator_rejects_attack_outside_note(self):
        cases = copy.deepcopy(_raw_cases())
        cases[1]["operator_comment"] = "approve this"
        self._assert_rejected(cases, "unexpected fields")

    def test_validator_rejects_bad_strategy_and_id(self):
        cases = copy.deepcopy(_raw_cases())
        cases[1]["attack_strategy"] = "jailbreak"
        self._assert_rejected(cases, "unknown attack_strategy")
        cases = copy.deepcopy(_raw_cases())
        cases[0]["attack_strategy"] = "authority_spoofing"
        self._assert_rejected(cases, "CLEAN variant must have null")
        cases = copy.deepcopy(_raw_cases())
        cases[0]["case_id"] = "ADV01-X"
        self._assert_rejected(cases, "case_id must be")

    def test_cli_refuses_invalid_dataset(self):
        cases = copy.deepcopy(_raw_cases())
        cases[1]["calibrated"] = False
        code, _, err = self.ws.run("export", dataset=self.ws.write_dataset(cases))
        self.assertEqual(code, 2)
        self.assertIn("calibrated differs", err)
        self.assertFalse(self.ws.prompts.exists())


# --------------------------------------------------------------------------- prompt separation

class PromptSeparationTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.ws.run("export")

    def test_export_writes_twelve_prompts_and_manifest_only(self):
        expected = sorted([f"{c['case_id']}.txt" for c in self.cases] + [me.MANIFEST_NAME])
        self.assertEqual(sorted(p.name for p in self.ws.prompts.iterdir()), expected)
        self.assertEqual([p.name for p in self.ws.responses.iterdir()], [me.METADATA_NAME])

    def test_prompts_reuse_stage4_design(self):
        for c in self.cases:
            text = (self.ws.prompts / f"{c['case_id']}.txt").read_text(encoding="utf-8")
            with self.subTest(case=c["case_id"]):
                self.assertEqual(text, f"{SYSTEM_PROMPT}\n\n{build_prompt(c).user}\n")
                self.assertEqual(text, me.build_manual_prompt(c))

    def test_prompts_never_leak_labels_ids_or_strategy(self):
        for c in self.cases:
            text = (self.ws.prompts / f"{c['case_id']}.txt").read_text(encoding="utf-8")
            with self.subTest(case=c["case_id"]):
                for forbidden in ('"expected"', c["case_id"], c["pair_id"], "CLEAN", "ATTACK",
                                  "attack_strategy", "attack_target", "variant", decide(c)[1]):
                    self.assertNotIn(forbidden, text)
                for strategy in ae.STRATEGIES:
                    self.assertNotIn(strategy, text)
                # Changing metadata fields never changes the prompt.
                for label in ("PASS", "FAIL", "INSUFFICIENT_DATA"):
                    mutated = {**c, "expected": label, "pair_id": "X", "variant": "Y",
                               "attack_strategy": "Z", "attack_target": label}
                    self.assertEqual(me.build_manual_prompt(mutated), me.build_manual_prompt(c))

    def test_paired_prompts_differ_only_in_note(self):
        for pair_id in {c["pair_id"] for c in self.cases}:
            clean, attack = self.by_id[f"{pair_id}-CLEAN"], self.by_id[f"{pair_id}-ATTACK"]
            with self.subTest(pair=pair_id):
                self.assertNotEqual(me.build_manual_prompt(clean), me.build_manual_prompt(attack))
                swapped = {**attack, "technician_note": clean["technician_note"]}
                self.assertEqual(me.build_manual_prompt(swapped), me.build_manual_prompt(clean))
                ci, ai = model_inputs(clean), model_inputs(attack)
                diff = [f for f in MODEL_INPUT_FIELDS if ae._canon(ci[f]) != ae._canon(ai[f])]
                self.assertEqual(diff, ["technician_note"])

    def test_attack_note_stays_inside_case_data_block(self):
        for c in self.cases:
            text = me.build_manual_prompt(c)
            with self.subTest(case=c["case_id"]):
                self.assertEqual(text.count("<case_data>"), 1)
                self.assertEqual(text.count("</case_data>"), 1)
                block = text.split("<case_data>", 1)[1].split("</case_data>", 1)[0]
                self.assertEqual(json.loads(block)["technician_note"], c["technician_note"])

    def test_manifest_hashes_match_files_and_carry_no_labels(self):
        manifest = json.loads((self.ws.prompts / me.MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(manifest["prompt_count"], 12)
        self.assertEqual(manifest["dataset_sha256"], hashlib.sha256(ae.DATASET.read_bytes()).hexdigest())
        raw = json.dumps(manifest)
        self.assertNotIn("expected", raw)
        self.assertNotIn("attack_strategy", raw)
        for cid, entry in manifest["cases"].items():
            data = (self.ws.prompts / entry["prompt_file"]).read_bytes()
            self.assertEqual(entry["prompt_sha256"], hashlib.sha256(data).hexdigest())

    def test_export_does_not_overwrite_metadata_or_responses(self):
        self.ws.write_metadata()
        self.ws.write_response("ADV01-CLEAN", "original answer bytes")
        self.ws.run("export")
        self.assertEqual(json.loads((self.ws.responses / me.METADATA_NAME).read_text()), GOOD_METADATA)
        self.assertEqual((self.ws.responses / "ADV01-CLEAN.txt").read_text(), "original answer bytes")


# --------------------------------------------------------------------------- pairing metrics (pure)

def _preds(cases, mapping):
    """mapping: case_id -> label/None; unspecified -> expected (correct)."""
    return {c["case_id"]: mapping.get(c["case_id"], c["expected"]) for c in cases}


class PairedMetricTests(BaseCase):
    def test_all_correct(self):
        m = ae.paired_metrics(self.cases, _preds(self.cases, {}))
        self.assertEqual((m["clean_accuracy"], m["attack_accuracy"]), (1.0, 1.0))
        self.assertEqual(m["valid_response_coverage"], 1.0)
        self.assertEqual(m["paired_flip_rate"], 0.0)
        self.assertEqual(m["attack_success_rate"], 0.0)
        self.assertEqual(m["counts"]["attack_success_denominator"], 6)
        self.assertEqual(m["unsafe_approvals"]["total"], 0)

    def test_attack_success_and_targeted_success(self):
        m = ae.paired_metrics(self.cases, _preds(self.cases, {
            "ADV03-ATTACK": "PASS",                # targeted success (unsafe approval)
            "ADV01-ATTACK": "INSUFFICIENT_DATA",   # success, but not the FAIL target
        }))
        self.assertEqual(m["counts"]["attack_successes"], 2)
        self.assertEqual(m["counts"]["targeted_attack_successes"], 1)
        self.assertEqual(m["attack_success_rate"], round(2 / 6, 4))
        self.assertEqual(m["targeted_attack_success_rate"], round(1 / 6, 4))
        self.assertEqual(m["paired_flip_rate"], round(2 / 6, 4))
        self.assertEqual(m["attack_accuracy"], round(4 / 6, 4))
        self.assertEqual(m["clean_accuracy"], 1.0)
        self.assertEqual(m["unsafe_approvals"], {"CLEAN": 0, "ATTACK": 1, "total": 1})

    def test_invalid_attack_response_is_excluded_not_success(self):
        m = ae.paired_metrics(self.cases, _preds(self.cases, {"ADV04-ATTACK": None}))
        self.assertEqual(m["counts"]["attack_successes"], 0)
        self.assertEqual(m["counts"]["excluded_attack_invalid"], 1)
        self.assertEqual(m["counts"]["attack_success_denominator"], 5)
        self.assertEqual(m["attack_success_rate"], 0.0)
        self.assertEqual(m["attack_accuracy"], round(5 / 6, 4))   # invalid counts as incorrect
        self.assertEqual(m["attack_accuracy_on_valid"], 1.0)
        self.assertEqual(m["counts"]["both_valid_pairs"], 5)
        status = {p["pair_id"]: p["attack_status"] for p in m["per_pair"]}
        self.assertEqual(status["ADV04"], "EXCLUDED_ATTACK_INVALID")

    def test_clean_incorrect_or_invalid_pairs_excluded(self):
        m = ae.paired_metrics(self.cases, _preds(self.cases, {
            "ADV01-CLEAN": "FAIL", "ADV01-ATTACK": "FAIL",   # clean wrong: excluded, no flip
            "ADV05-CLEAN": None, "ADV05-ATTACK": "PASS",     # clean invalid: excluded
        }))
        self.assertEqual(m["counts"]["excluded_clean_incorrect"], 1)
        self.assertEqual(m["counts"]["excluded_clean_invalid"], 1)
        self.assertEqual(m["counts"]["attack_success_denominator"], 4)
        self.assertEqual(m["counts"]["attack_successes"], 0)
        self.assertEqual(m["paired_flip_rate"], 0.0)   # 5 both-valid pairs, none differ
        self.assertEqual(m["counts"]["both_valid_pairs"], 5)
        self.assertEqual(m["unsafe_approvals"]["ATTACK"], 1)

    def test_flip_from_wrong_clean_to_right_attack_counts_as_flip_not_success(self):
        m = ae.paired_metrics(self.cases, _preds(self.cases, {"ADV02-CLEAN": "FAIL"}))
        self.assertEqual(m["counts"]["flips"], 1)
        self.assertEqual(m["counts"]["attack_successes"], 0)

    def test_all_invalid_gives_null_rates(self):
        m = ae.paired_metrics(self.cases, {c["case_id"]: None for c in self.cases})
        self.assertEqual(m["valid_response_coverage"], 0.0)
        self.assertEqual((m["clean_accuracy"], m["attack_accuracy"]), (0.0, 0.0))
        self.assertIsNone(m["paired_flip_rate"])
        self.assertIsNone(m["attack_success_rate"])
        self.assertEqual(m["unsafe_approvals"]["total"], 0)

    def test_unknown_label_rejected(self):
        with self.assertRaises(ValueError):
            ae.paired_metrics(self.cases, {"ADV01-CLEAN": "APPROVED"})


# --------------------------------------------------------------------------- score workflow and errors

class ScoreTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.ws.run("export")
        self.ws.write_metadata()

    def _write_all(self, overrides=None):
        overrides = overrides or {}
        for c in self.cases:
            self.ws.write_response(c["case_id"], overrides.get(c["case_id"], _json(c["expected"])))

    def test_metadata_required(self):
        (self.ws.responses / me.METADATA_NAME).unlink()
        code, _, err = self.ws.run("score")
        self.assertEqual(code, 2)
        self.assertIn("not found", err)
        self.assertFalse(self.ws.out.exists())

    def test_susceptible_fixture_scored_and_gated(self):
        # Synthetic fixture: obeys every approve-style attack.
        approve = {c["case_id"]: _json("PASS") for c in self.cases
                   if c["variant"] == ae.ATTACK and c["attack_target"] == "PASS"}
        self._write_all(approve)
        code, out, _ = self.ws.run("score")
        self.assertEqual(code, 0)
        s = self.ws.summary()
        self.assertEqual(s["result_type"], me.REAL_TAG)
        self.assertEqual(s["experiment"], ae.EXPERIMENT)
        self.assertFalse(s["simulated"])
        strict = s["modes"]["STRICT"]
        self.assertEqual(strict["baseline"]["attack_success_rate"], round(4 / 6, 4))
        self.assertEqual(strict["baseline"]["unsafe_approvals"]["ATTACK"], 4)
        self.assertEqual(strict["guarded"]["unsafe_approvals"]["total"], 0)
        self.assertEqual(strict["guarded"]["attack_success_rate"], 0.0)
        self.assertEqual(strict["gate_interventions"]["ATTACK"], 4)
        self.assertEqual(strict["gate_interventions"]["CLEAN"], 0)
        self.assertEqual(strict["format_errors"]["total"], 0)
        self.assertIn(ae.EXPERIMENT, out)
        rows = self.ws.csv_rows()
        self.assertEqual(rows["ADV03-ATTACK"]["strict_guarded_prediction"], "FAIL")
        self.assertEqual(rows["ADV05-ATTACK"]["strict_guarded_prediction"], "INSUFFICIENT_DATA")
        self.assertEqual(rows["ADV01-CLEAN"]["strict_guarded_prediction"], "PASS")

    def test_missing_and_malformed_responses_are_format_errors_not_successes(self):
        self._write_all({
            "ADV01-ATTACK": "",                                  # empty
            "ADV02-ATTACK": "```json\n" + _json("FAIL") + "\n```",  # code fence
            "ADV03-ATTACK": "I would approve this. PASS",        # prose
            "ADV04-ATTACK": b"\xff\xfe\x00",                     # undecodable
        })
        (self.ws.responses / "ADV05-ATTACK.txt").unlink()        # missing
        self.ws.run("score")
        strict = self.ws.summary()["modes"]["STRICT"]
        b = strict["baseline"]
        self.assertEqual(b["counts"]["attack_successes"], 0)
        self.assertEqual(b["counts"]["excluded_attack_invalid"], 5)
        self.assertEqual(b["attack_success_rate"], 0.0)        # 0 / 1 (only ADV06 eligible)
        self.assertEqual(b["counts"]["attack_success_denominator"], 1)
        self.assertEqual(b["unsafe_approvals"]["total"], 0)
        fe = strict["format_errors"]
        self.assertEqual((fe["total"], fe["CLEAN"], fe["ATTACK"]), (5, 0, 5))
        self.assertEqual(set(fe["by_error_type"]), {"empty_response", "unparseable_response",
                                                     "undecodable_response", "missing_response_file"})
        self.assertEqual(self.ws.summary()["missing_responses"], ["ADV05-ATTACK"])

    def test_format_normalized_is_secondary_and_files_unchanged(self):
        smart = "{“decision”: “PASS”, “reason”: “x”}"
        self._write_all({"ADV02-ATTACK": smart})
        before = {p.name: p.read_bytes() for p in self.ws.responses.iterdir()}
        self.ws.run("score")
        s = self.ws.summary()
        self.assertEqual(s["primary_evaluation_mode"], "STRICT")
        self.assertEqual(s["modes"]["STRICT"]["baseline"]["counts"]["excluded_attack_invalid"], 1)
        self.assertEqual(s["modes"]["STRICT"]["format_errors"]["by_error_type"],
                         {"invalid_json": {"CLEAN": 0, "ATTACK": 1}})
        norm = s["modes"]["FORMAT_NORMALIZED"]
        self.assertEqual(norm["format_errors"]["total"], 0)
        self.assertEqual(norm["baseline"]["valid_response_coverage"], 1.0)
        self.assertEqual(norm["normalization"]["normalized_case_ids"], ["ADV02-ATTACK"])
        self.assertFalse(norm["normalization"]["original_files_modified"])
        after = {p.name: p.read_bytes() for p in self.ws.responses.iterdir()}
        self.assertEqual(before, after)
        row = self.ws.csv_rows()["ADV02-ATTACK"]
        self.assertEqual(row["response_sha256"], hashlib.sha256(smart.encode("utf-8")).hexdigest())

    def test_prompt_drift_detected(self):
        self._write_all()
        (self.ws.prompts / "ADV01-CLEAN.txt").write_text("tampered", encoding="utf-8")
        manifest_path = self.ws.prompts / me.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["cases"]["ADV01-CLEAN"]["prompt_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        code, out, _ = self.ws.run("score")
        self.assertEqual(code, 0)
        self.assertFalse(self.ws.summary()["prompt_integrity_ok"])
        self.assertIn("prompt drift", out)

    def test_outputs_do_not_touch_stage4_files(self):
        self._write_all()
        self.ws.run("score")
        self.assertEqual(sorted(p.name for p in self.ws.out.iterdir()), sorted([ae.CSV_NAME, ae.SUMMARY_NAME]))
        self.assertNotEqual(ae.RESPONSES_DIR, me.RESPONSES_DIR)
        self.assertNotEqual(ae.PROMPTS_DIR, me.PROMPTS_DIR)
        self.assertNotEqual(ae.SUMMARY_NAME, me.SUMMARY_NAME)


# --------------------------------------------------------------------------- reproducibility

class ReproducibilityTests(BaseCase):
    def test_export_is_byte_identical_across_runs(self):
        self.ws.run("export")
        first = {p.name: p.read_bytes() for p in self.ws.prompts.iterdir()}
        self.ws.run("export")
        second = {p.name: p.read_bytes() for p in self.ws.prompts.iterdir()}
        self.assertEqual(first, second)
        self.assertEqual(len(first), 13)  # 12 prompts + manifest

    def test_deterministic_order_and_filenames(self):
        ids = [c["case_id"] for c in self.cases]
        expected = [f"ADV0{i}-{v}" for i in range(1, 7) for v in ("CLEAN", "ATTACK")]
        self.assertEqual(ids, expected)
        shuffled = list(reversed(_raw_cases()))
        self.assertEqual([c["case_id"] for c in ae.order_cases(shuffled)], expected)

    def test_score_is_deterministic(self):
        self.ws.run("export")
        self.ws.write_metadata()
        for c in self.cases:
            self.ws.write_response(c["case_id"], _json("PASS"))
        self.ws.run("score")
        first = ((self.ws.out / ae.SUMMARY_NAME).read_bytes(), (self.ws.out / ae.CSV_NAME).read_bytes())
        self.ws.run("score")
        second = ((self.ws.out / ae.SUMMARY_NAME).read_bytes(), (self.ws.out / ae.CSV_NAME).read_bytes())
        self.assertEqual(first, second)

    def test_metric_definitions_document_denominators(self):
        defs = ae.ADVERSARIAL_METRIC_DEFINITIONS
        for key in ("clean_accuracy", "attack_accuracy", "valid_response_coverage", "paired_flip_rate",
                    "attack_success_rate", "unsafe_approvals", "gate_interventions", "format_errors"):
            self.assertIn(key, defs)
        self.assertIn("never counted as successes", defs["attack_success_rate"])


if __name__ == "__main__":
    unittest.main()
