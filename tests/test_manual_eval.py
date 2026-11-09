"""Tests for the manual evaluation workflow.

All "responses" here are SYNTHETIC FIXTURES written by the tests into temporary
directories. No model is queried and the real manual_responses/ is never touched.
"""
import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from src import manual_eval as me
from src.evaluate import ROOT, load_cases
from src.llm_baseline import SIMULATED_TAG, SYSTEM_PROMPT, build_prompt, decide

DATASET = ROOT / "data" / "cases.jsonl"
IDS = ["TC001", "TC004", "TC011"]
GOOD_METADATA = {
    "model_name": "synthetic-fixture-model",
    "experiment_date": "2026-11-09",
    "interface": "synthetic test fixture (no chat used)",
    "chat_settings": "n/a (unit test)",
    "operator_limitations": "",
}


def _json(decision):
    return json.dumps({"decision": decision, "reason": "synthetic fixture"})


class Workspace:
    """Temporary prompts/responses/results directories."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.prompts, self.responses, self.out = base / "prompts", base / "responses", base / "results"

    def close(self):
        self._tmp.cleanup()

    def run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        common = ["--case-ids", ",".join(IDS), "--prompts-dir", str(self.prompts),
                  "--responses-dir", str(self.responses)]
        if argv[0] == "score":
            common += ["--out", str(self.out)]
        with redirect_stdout(out), redirect_stderr(err):
            code = me.main([*argv, *common])
        return code, out.getvalue(), err.getvalue()

    def write_metadata(self, metadata=GOOD_METADATA):
        self.responses.mkdir(parents=True, exist_ok=True)
        (self.responses / me.METADATA_NAME).write_text(json.dumps(metadata), encoding="utf-8")

    def write_response(self, cid, text):
        self.responses.mkdir(parents=True, exist_ok=True)
        (self.responses / f"{cid}.txt").write_bytes(text.encode("utf-8") if isinstance(text, str) else text)

    def summary(self):
        return json.loads((self.out / me.SUMMARY_NAME).read_text(encoding="utf-8"))

    def csv_rows(self):
        with (self.out / me.CSV_NAME).open(encoding="utf-8", newline="") as handle:
            return {row["case_id"]: row for row in csv.DictReader(handle)}


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.close)
        self.cases = {c["case_id"]: c for c in load_cases(DATASET)}


class ExportTests(BaseCase):
    def test_export_writes_one_prompt_per_case_and_no_responses(self):
        code, out, _ = self.ws.run("export")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in self.ws.prompts.iterdir()),
                         sorted([f"{c}.txt" for c in IDS] + [me.MANIFEST_NAME]))
        # Export must never create (fabricate) response files.
        self.assertEqual([p.name for p in self.ws.responses.iterdir()], [me.METADATA_NAME])

    def test_prompt_reuses_baseline_design_and_is_self_contained(self):
        self.ws.run("export")
        for cid in IDS:
            text = (self.ws.prompts / f"{cid}.txt").read_text(encoding="utf-8")
            with self.subTest(case_id=cid):
                self.assertEqual(text, f"{SYSTEM_PROMPT}\n\n{build_prompt(self.cases[cid]).user}\n")
                self.assertIn("<= 40.0 C", text)
                self.assertIn("<= 2.0 g", text)
                self.assertIn("untrusted", text)
                self.assertIn(json.dumps(self.cases[cid]["technician_note"])[1:-1], text)

    def test_exported_prompts_never_expose_labels_or_answers(self):
        self.ws.run("export")
        exported = list(self.ws.prompts.iterdir())
        for path in exported:
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name):
                self.assertNotIn('"expected"', text)
                self.assertNotIn("reference", text.lower())
                if path.name != me.MANIFEST_NAME:
                    cid = path.stem
                    self.assertNotIn(cid, text)
                    self.assertNotIn(decide(self.cases[cid])[1], text)
        for cid in IDS:
            for label in ("PASS", "FAIL", "INSUFFICIENT_DATA"):
                with self.subTest(case_id=cid, label=label):
                    self.assertEqual(me.build_manual_prompt(self.cases[cid]),
                                     me.build_manual_prompt({**self.cases[cid], "expected": label}))

    def test_export_does_not_overwrite_user_metadata(self):
        self.ws.write_metadata()
        self.ws.run("export")
        saved = json.loads((self.ws.responses / me.METADATA_NAME).read_text())
        self.assertEqual(saved, GOOD_METADATA)

    def test_unknown_case_id_rejected(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            me.main(["export", "--case-ids", "TC999", "--prompts-dir", str(self.ws.prompts),
                     "--responses-dir", str(self.ws.responses)])


class MetadataTests(BaseCase):
    def _score_code(self):
        return self.ws.run("score")

    def test_template_placeholders_block_scoring(self):
        self.ws.run("export")
        code, _, err = self._score_code()
        self.assertEqual(code, 2)
        self.assertIn("FILL_IN", err)
        self.assertFalse(self.ws.out.exists())

    def test_missing_metadata_blocks_scoring(self):
        code, _, err = self._score_code()
        self.assertEqual(code, 2)
        self.assertIn("not found", err)

    def test_invalid_metadata_values_rejected(self):
        bad = [
            {**GOOD_METADATA, "model_name": ""},
            {**GOOD_METADATA, "experiment_date": "09/11/2026"},
            {k: v for k, v in GOOD_METADATA.items() if k != "interface"},
            {**GOOD_METADATA, "chat_settings": "FILL_IN later"},
        ]
        for metadata in bad:
            with self.subTest(metadata=metadata):
                self.ws.write_metadata(metadata)
                with self.assertRaises(me.MetadataError):
                    me.load_metadata(self.ws.responses / me.METADATA_NAME)


class ScoreTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.ws.run("export")
        self.ws.write_metadata()

    def test_real_answers_scored_and_gated(self):
        # Synthetic fixture: a model that approves everything.
        for cid in IDS:
            self.ws.write_response(cid, _json("PASS") + "\n")
        code, out, _ = self.ws.run("score")
        self.assertEqual(code, 0)
        s = self.ws.summary()
        self.assertEqual(s["result_type"], me.REAL_TAG)
        self.assertNotEqual(s["result_type"], SIMULATED_TAG)
        self.assertFalse(s["simulated"])
        self.assertEqual(s["metadata"], GOOD_METADATA)
        self.assertEqual(s["baseline"]["unsafe_approvals"], 2)   # TC004, TC011
        self.assertEqual(s["guarded"]["unsafe_approvals"], 0)
        changed = {c["case_id"]: c["guarded_prediction"] for c in s["changed_cases"]}
        self.assertEqual(changed, {"TC004": "INSUFFICIENT_DATA", "TC011": "FAIL"})
        rows = self.ws.csv_rows()
        self.assertEqual(rows["TC001"]["guarded_prediction"], "PASS")
        self.assertTrue(all(r["result_type"] == me.REAL_TAG for r in rows.values()))
        self.assertTrue(all(r["model_name"] == "synthetic-fixture-model" for r in rows.values()))
        self.assertIn(me.REAL_TAG, out)
        self.assertTrue(s["methodology_limitations"])

    def test_correct_answers_unchanged_by_gate(self):
        for cid, label in (("TC001", "PASS"), ("TC004", "INSUFFICIENT_DATA"), ("TC011", "FAIL")):
            self.ws.write_response(cid, _json(label))
        self.ws.run("score")
        s = self.ws.summary()
        self.assertEqual(s["vetoed_count"], 0)
        self.assertEqual(s["baseline"]["accuracy"], 1.0)
        self.assertEqual(s["guarded"]["accuracy"], 1.0)

    def test_missing_response_files_are_errors_not_pass(self):
        self.ws.write_response("TC001", _json("PASS"))
        self.ws.run("score")
        s, rows = self.ws.summary(), self.ws.csv_rows()
        self.assertEqual(s["missing_responses"], ["TC004", "TC011"])
        for cid in ("TC004", "TC011"):
            with self.subTest(case_id=cid):
                self.assertEqual(rows[cid]["baseline_prediction"], "INVALID")
                self.assertEqual(rows[cid]["guarded_prediction"], "INVALID")
                self.assertEqual(rows[cid]["baseline_parse_error"], "missing_response_file")
        self.assertEqual(s["baseline"]["error_count"], 2)
        self.assertEqual(s["guarded"]["error_count"], 2)
        self.assertEqual(s["baseline"]["unsafe_approvals"], 0)

    def test_malformed_responses_remain_errors(self):
        malformed = {
            "```json\n" + _json("PASS") + "\n```",   # code fence: strict parser rejects
            "The unit should PASS.",
            "",
            '{"decision": "APPROVED"}',
            "﻿" + _json("PASS"),                 # BOM is not stripped silently
            "PASS\nFAIL",
        }
        for text in malformed:
            with self.subTest(text=text):
                for cid in IDS:
                    self.ws.write_response(cid, text)
                self.ws.run("score")
                s = self.ws.summary()
                self.assertEqual(s["baseline"]["error_count"], 3)
                self.assertEqual(s["guarded"]["error_count"], 3)
                self.assertEqual(s["baseline"]["unsafe_approvals"], 0)
                self.assertEqual(s["guarded"]["confusion_matrix"]["matrix"]["PASS"]["PASS"], 0)

    def test_undecodable_bytes_are_errors(self):
        for cid in IDS:
            self.ws.write_response(cid, b"\xff\xfe\x00P")
        self.ws.run("score")
        self.assertEqual(set(r["baseline_parse_error"] for r in self.ws.csv_rows().values()),
                         {"undecodable_response"})

    def test_raw_response_recorded_verbatim_with_hash(self):
        raw = '  {"decision": "FAIL", "reason": "Vibration 2.1 g > 2.0 g."}\r\n'
        for cid in IDS:
            self.ws.write_response(cid, raw)
        self.ws.run("score")
        row = self.ws.csv_rows()["TC011"]
        self.assertEqual(row["raw_response"], raw)
        self.assertEqual(row["response_sha256"], hashlib.sha256(raw.encode("utf-8")).hexdigest())
        self.assertEqual(row["baseline_prediction"], "FAIL")

    def test_same_answers_feed_both_arms(self):
        for cid, label in zip(IDS, ("FAIL", "PASS", "PASS")):
            self.ws.write_response(cid, _json(label))
        self.ws.run("score")
        for cid, row in self.ws.csv_rows().items():
            with self.subTest(case_id=cid):
                self.assertEqual(json.loads(row["raw_response"])["decision"], row["baseline_prediction"])
                if row["vetoed"] == "False":
                    self.assertEqual(row["guarded_prediction"], row["baseline_prediction"])

    def test_prompt_drift_is_detected(self):
        for cid in IDS:
            self.ws.write_response(cid, _json("FAIL"))
        manifest_path = self.ws.prompts / me.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["cases"]["TC004"]["prompt_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        self.ws.run("score")
        s, rows = self.ws.summary(), self.ws.csv_rows()
        self.assertFalse(s["prompt_integrity_ok"])
        self.assertEqual(rows["TC004"]["prompt_matches_export"], "False")
        self.assertEqual(rows["TC001"]["prompt_matches_export"], "True")

    def test_format_normalized_is_secondary_and_strict_is_preserved(self):
        # Synthetic smart-quote fixtures reproducing the observed pattern.
        L, R = "“", "”"
        fixtures = {cid: f"{{{L}decision{R}: {L}{label}{R}, {L}reason{R}: {L}synthetic note’s{R}}}"
                    for cid, label in zip(IDS, ("PASS", "INSUFFICIENT_DATA", "FAIL"))}
        for cid, text in fixtures.items():
            self.ws.write_response(cid, text)
        before = {cid: (self.ws.responses / f"{cid}.txt").read_bytes() for cid in IDS}
        self.ws.run("score")
        s, rows = self.ws.summary(), self.ws.csv_rows()

        # STRICT (primary) results are unchanged: all invalid.
        self.assertEqual(s["primary_evaluation_mode"], "STRICT")
        self.assertEqual(s["baseline"]["coverage"], 0.0)
        self.assertEqual(s["baseline"]["error_count"], 3)
        self.assertEqual(s["invalid_responses"], IDS)
        self.assertTrue(all(r["baseline_prediction"] == "INVALID" for r in rows.values()))

        # FORMAT_NORMALIZED (secondary) reported separately.
        fn = s["format_normalized"]
        self.assertEqual(fn["evaluation_mode"], "FORMAT_NORMALIZED")
        self.assertEqual(fn["baseline"]["coverage"], 1.0)
        self.assertEqual(fn["baseline"]["accuracy"], 1.0)
        self.assertEqual(fn["normalized_case_ids"], IDS)
        self.assertEqual(s["mode_comparison"]["STRICT"]["coverage"]["baseline"], 0.0)
        self.assertEqual(s["mode_comparison"]["FORMAT_NORMALIZED"]["coverage"]["baseline"], 1.0)
        self.assertFalse(fn["original_files_modified"])

        # Original files and hashes preserved; transformations recorded per case.
        for cid in IDS:
            with self.subTest(case_id=cid):
                self.assertEqual((self.ws.responses / f"{cid}.txt").read_bytes(), before[cid])
                original = hashlib.sha256(before[cid]).hexdigest()
                self.assertEqual(rows[cid]["response_sha256"], original)
                self.assertEqual(fn["per_case"][cid]["original_response_sha256"], original)
                self.assertNotEqual(fn["per_case"][cid]["normalized_sha256"], original)
                self.assertEqual(rows[cid]["raw_response"], fixtures[cid])
                self.assertEqual(rows[cid]["normalization_transformations"], "U+201C->U+0022 x4; U+201D->U+0022 x4")
                self.assertEqual(rows[cid]["normalized_prediction"], rows[cid]["expected"])

    def test_format_normalized_keeps_missing_and_unfixable_as_errors(self):
        L, R = "“", "”"
        self.ws.write_response("TC001", f"{{{L}decision{R}: {L}PASS{R}")       # truncated
        self.ws.write_response("TC004", "```json\n" + _json("PASS") + "\n```")  # fenced
        # TC011 missing
        self.ws.run("score")
        fn = self.ws.summary()["format_normalized"]
        self.assertEqual(fn["still_invalid_case_ids"], IDS)
        self.assertEqual(fn["baseline"]["error_count"], 3)
        self.assertEqual(fn["baseline"]["unsafe_approvals"], 0)
        self.assertEqual(fn["per_case"]["TC011"]["normalized_parse_error"], "missing_response_file")
        self.assertEqual(fn["per_case"]["TC004"]["normalized_parse_error"], "unparseable_response")
        self.assertTrue(fn["per_case"]["TC001"]["normalized_parse_error"].startswith("still_invalid"))

    def test_scoring_never_uses_a_mock_or_model_client(self):
        for cid in IDS:
            self.ws.write_response(cid, _json("PASS"))
        with mock.patch("src.llm_baseline.MockClient.complete", side_effect=AssertionError("model called")), \
             mock.patch("src.llm_baseline.AnthropicClient.__init__", side_effect=AssertionError("api client")):
            code, _, _ = self.ws.run("score")
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
