import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src import compare
from src.evaluate import ROOT, load_cases
from src.llm_baseline import SIMULATED_TAG, MockClient, predict

DATASET = ROOT / "data" / "cases.jsonl"


class CountingClient:
    """Wraps a client and records every prompt it receives."""

    simulated = True

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.inner.complete(prompt)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.cases = load_cases(DATASET)
        self.client = CountingClient(MockClient("injection_susceptible"))
        self.rows, self.summary = compare.run_comparison(self.cases, self.client)
        self.by_id = {r["case_id"]: r for r in self.rows}

    def test_model_called_exactly_once_per_case(self):
        self.assertEqual(len(self.client.prompts), len(self.cases))
        self.assertEqual(self.summary["model_calls"], len(self.cases))

    def test_identical_baseline_predictions_feed_both_arms(self):
        independent = predict(self.cases, MockClient("injection_susceptible"))
        self.assertEqual([r["baseline_prediction"] for r in self.rows],
                         [p["predicted"] for p in independent])
        # Guarded arm differs from baseline only where a veto happened.
        for r in self.rows:
            if not r["vetoed"]:
                self.assertEqual(r["guarded_prediction"], r["baseline_prediction"])
            else:
                self.assertEqual(r["baseline_prediction"], "PASS")

    def test_gate_reuses_given_predictions_without_model_calls(self):
        predictions = [{"case_id": c["case_id"], "predicted": "PASS", "parse_error": None, "raw_response": "PASS"}
                       for c in self.cases]
        gated = compare.apply_gate_to_predictions(predictions, self.cases)
        self.assertTrue(all(r["baseline_prediction"] == "PASS" for r in gated))

    def test_same_cases_and_labels_for_both_arms(self):
        self.assertEqual(self.summary["baseline"]["cases"], self.summary["guarded"]["cases"])
        self.assertEqual(self.summary["baseline"]["label_counts"], self.summary["guarded"]["label_counts"])

    def test_expected_scenarios(self):
        expectations = {
            "TC011": ("PASS", "FAIL", True),
            "TC008": ("PASS", "FAIL", True),
            "TC004": ("PASS", "INSUFFICIENT_DATA", True),
            "TC007": ("FAIL", "FAIL", False),      # one-sided: false rejection is not repaired
            "TC001": ("PASS", "PASS", False),
            "TC005": ("INSUFFICIENT_DATA", "INSUFFICIENT_DATA", False),
        }
        for cid, (base, guarded, vetoed) in expectations.items():
            with self.subTest(case_id=cid):
                r = self.by_id[cid]
                self.assertEqual((r["baseline_prediction"], r["guarded_prediction"], r["vetoed"]),
                                 (base, guarded, vetoed))
        self.assertEqual(self.summary["vetoed_count"], 3)
        self.assertEqual({c["case_id"] for c in self.summary["changed_cases"]}, {"TC004", "TC008", "TC011"})

    def test_guarded_arm_removes_unsafe_approvals(self):
        b, g = self.summary["baseline"], self.summary["guarded"]
        self.assertGreater(b["unsafe_approvals"], 0)
        self.assertEqual(g["false_approvals"], 0)
        self.assertEqual(g["unsafe_approvals"], 0)
        self.assertEqual(g["false_rejections"], b["false_rejections"])

    def test_labels_do_not_influence_gate(self):
        predictions = predict(self.cases, MockClient("injection_susceptible"))
        relabelled = [{**c, "expected": "PASS"} for c in self.cases]
        a = compare.apply_gate_to_predictions(predictions, self.cases)
        b = compare.apply_gate_to_predictions(predictions, relabelled)
        self.assertEqual(a, b)
        self.assertNotIn("expected", a[0])

    def test_gate_inputs_projection(self):
        projected = compare.gate_inputs(self.cases[0])
        self.assertEqual(set(projected), {"temperature_c", "vibration_g", "calibrated"})

    def test_invalid_responses_remain_errors_in_both_arms(self):
        _, summary = compare.run_comparison(self.cases, MockClient("malformed"))
        self.assertGreater(summary["baseline"]["error_count"], 0)
        self.assertEqual(summary["guarded"]["error_count"], summary["baseline"]["error_count"])
        self.assertEqual(summary["guarded"]["coverage"], summary["baseline"]["coverage"])

    def test_summary_is_labelled_simulated(self):
        self.assertTrue(self.summary["simulated"])
        self.assertEqual(self.summary["result_type"], SIMULATED_TAG)
        self.assertIn("NOT evidence", self.summary["warning"])
        self.assertIn("unsafe_approval_rate", self.summary["metric_definitions"])


class CompareCliTests(unittest.TestCase):
    def test_cli_writes_labelled_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = compare.main(["--mock-profile", "injection_susceptible", "--out", tmp])
            self.assertEqual(code, 0)
            summary = json.loads((Path(tmp) / "comparison_summary.json").read_text())
            lines = (Path(tmp) / "compare.csv").read_text().strip().splitlines()
        self.assertEqual(summary["result_type"], SIMULATED_TAG)
        self.assertEqual(len(lines) - 1, 11)
        self.assertTrue(all(line.startswith(SIMULATED_TAG) for line in lines[1:]))
        self.assertIn(SIMULATED_TAG, out.getvalue())
        self.assertIn("TC011: PASS -> FAIL", out.getvalue())


if __name__ == "__main__":
    unittest.main()
