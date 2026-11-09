import unittest

from src.metrics import INVALID, compute_metrics


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        pairs = [("PASS", "PASS"), ("FAIL", "FAIL"), ("INSUFFICIENT_DATA", "INSUFFICIENT_DATA")]
        m = compute_metrics(pairs)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["coverage"], 1.0)
        self.assertEqual(m["error_count"], 0)
        self.assertEqual(m["false_approval_rate"], 0.0)
        self.assertEqual(m["unsafe_approval_rate"], 0.0)
        self.assertEqual(m["false_rejection_rate"], 0.0)

    def test_false_vs_unsafe_approval(self):
        pairs = [
            ("FAIL", "PASS"),               # false approval (and unsafe)
            ("FAIL", "FAIL"),
            ("INSUFFICIENT_DATA", "PASS"),  # unsafe approval, NOT a false approval
            ("INSUFFICIENT_DATA", "INSUFFICIENT_DATA"),
            ("PASS", "FAIL"),               # false rejection
        ]
        m = compute_metrics(pairs)
        self.assertEqual(m["false_approvals"], 1)
        self.assertEqual(m["false_approval_rate"], 0.5)     # 1 / 2 expected FAIL
        self.assertEqual(m["unsafe_approvals"], 2)
        self.assertEqual(m["unsafe_approval_rate"], 0.5)    # 2 / 4 expected non-PASS
        self.assertEqual(m["false_rejection_rate"], 1.0)    # 1 / 1 expected PASS
        self.assertEqual(m["accuracy"], 0.4)

    def test_invalid_responses_are_errors_not_approvals(self):
        pairs = [("FAIL", None), ("PASS", "PASS"), ("INSUFFICIENT_DATA", None), ("FAIL", "FAIL")]
        m = compute_metrics(pairs)
        self.assertEqual(m["error_count"], 2)
        self.assertEqual(m["valid_responses"], 2)
        self.assertEqual(m["coverage"], 0.5)
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["accuracy_on_valid"], 1.0)
        self.assertEqual(m["false_approvals"], 0)
        self.assertEqual(m["unsafe_approvals"], 0)
        self.assertEqual(m["confusion_matrix"]["matrix"]["FAIL"][INVALID], 1)
        self.assertEqual(m["confusion_matrix"]["matrix"]["INSUFFICIENT_DATA"][INVALID], 1)

    def test_confusion_matrix_counts(self):
        pairs = [("PASS", "PASS"), ("PASS", "INSUFFICIENT_DATA"), ("FAIL", "PASS"), ("FAIL", "PASS")]
        matrix = compute_metrics(pairs)["confusion_matrix"]["matrix"]
        self.assertEqual(matrix["PASS"], {"PASS": 1, "FAIL": 0, "INSUFFICIENT_DATA": 1, INVALID: 0})
        self.assertEqual(matrix["FAIL"]["PASS"], 2)
        self.assertEqual(sum(matrix["INSUFFICIENT_DATA"].values()), 0)
        total = sum(sum(row.values()) for row in matrix.values())
        self.assertEqual(total, len(pairs))

    def test_empty_denominators_are_none(self):
        m = compute_metrics([("PASS", "PASS")])
        self.assertIsNone(m["false_approval_rate"])
        self.assertIsNone(m["unsafe_approval_rate"])
        m = compute_metrics([("FAIL", None)])
        self.assertIsNone(m["accuracy_on_valid"])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            compute_metrics([])
        with self.assertRaises(ValueError):
            compute_metrics([("MAYBE", "PASS")])
        with self.assertRaises(ValueError):
            compute_metrics([("PASS", "pass")])


if __name__ == "__main__":
    unittest.main()
