"""Regression tests for the secondary FORMAT_NORMALIZED parser.

Fixtures are SYNTHETIC strings that reproduce the pattern observed in the first
manual run (every JSON double quote was U+201C/U+201D). They are not model output.
"""
import json
import unittest

from src.llm_baseline import parse_response
from src.normalization import (NORMALIZATION_VERSION, TYPOGRAPHIC_DOUBLE_QUOTES,
                               format_normalized_parse, normalize_typographic_double_quotes)

L, R = "“", "”"   # LEFT / RIGHT DOUBLE QUOTATION MARK
APOS = "’"            # RIGHT SINGLE QUOTATION MARK


def smart(decision, reason):
    """Synthetic smart-quoted JSON object, same shape as the observed responses."""
    return f"{{{L}decision{R}: {L}{decision}{R}, {L}reason{R}: {L}{reason}{R}}}"


class NormalizeFunctionTests(unittest.TestCase):
    def test_only_typographic_double_quotes_are_replaced(self):
        raw = smart("INSUFFICIENT_DATA", f"Synthetic fixture text; the operator{APOS}s remark is kept.")
        normalized, transformations = normalize_typographic_double_quotes(raw)
        self.assertEqual(len(normalized), len(raw))
        for a, b in zip(raw, normalized):
            if a in TYPOGRAPHIC_DOUBLE_QUOTES:
                self.assertEqual(b, '"')
            else:
                self.assertEqual(a, b)          # every other character identical
        self.assertIn(APOS, normalized)         # apostrophe U+2019 untouched

    def test_transformations_record_codepoints_counts_positions(self):
        raw = smart("FAIL", "Synthetic fixture reason 7.5 > 1.0.")
        _, transformations = normalize_typographic_double_quotes(raw)
        by_from = {t["from"]: t for t in transformations}
        self.assertEqual(set(by_from), {"U+201C", "U+201D"})
        self.assertEqual(by_from["U+201C"]["count"], 4)
        self.assertEqual(by_from["U+201D"]["count"], 4)
        self.assertEqual(by_from["U+201C"]["positions"][0], 1)
        self.assertEqual(by_from["U+201D"]["from_name"], "RIGHT DOUBLE QUOTATION MARK")
        self.assertTrue(all(t["to"] == "U+0022" and t["rule"] == NORMALIZATION_VERSION for t in transformations))

    def test_low_and_reversed_quotes_supported(self):
        raw = "{„decision‟: „PASS‟}"
        self.assertEqual(format_normalized_parse(raw).decision, "PASS")


class FormatNormalizedParseTests(unittest.TestCase):
    def test_smart_quoted_json_strict_invalid_normalized_valid(self):
        for label in ("PASS", "FAIL", "INSUFFICIENT_DATA"):
            raw = smart(label, "Synthetic reason.")
            with self.subTest(label=label):
                self.assertEqual(parse_response(raw).error, "invalid_json")   # STRICT unchanged
                result = format_normalized_parse(raw)
                self.assertEqual(result.decision, label)
                self.assertTrue(result.applied and result.eligible)
                self.assertIsNone(result.error)

    def test_content_is_preserved_after_normalization(self):
        reason = f"Synthetic fixture: reading 9.75 vs limit 1.5; operator{APOS}s remark kept."
        normalized, _ = normalize_typographic_double_quotes(smart("FAIL", reason))
        self.assertEqual(json.loads(normalized), {"decision": "FAIL", "reason": reason})

    def test_strictly_valid_response_is_not_transformed(self):
        raw = json.dumps({"decision": "PASS", "reason": f"note{APOS}s text"})
        result = format_normalized_parse(raw)
        self.assertEqual(result.decision, "PASS")
        self.assertFalse(result.applied)
        self.assertEqual(result.transformations, ())

    def test_unknown_labels_are_never_mapped_or_guessed(self):
        for label in ("APPROVED", "PASS or FAIL", "OK", "FAILED"):
            with self.subTest(label=label):
                result = format_normalized_parse(smart(label, "synthetic"))
                self.assertIsNone(result.decision)
                self.assertEqual(result.error, "still_invalid_after_normalization:unknown_label")

    def test_label_case_rule_is_inherited_from_strict_parser(self):
        # The strict parser already accepts " fail " for straight-quoted JSON;
        # normalization adds no label rule of its own.
        self.assertEqual(parse_response('{"decision": "fail"}').decision, "FAIL")
        self.assertEqual(format_normalized_parse(smart("fail", "x")).decision, "FAIL")

    def test_still_invalid_after_normalization_stays_invalid(self):
        still_invalid = [
            f"{{{L}decision{R}: {L}PASS{R}",                        # missing closing brace: not guessed
            f"{{{L}decision{R}: {L}PASS{R}, {L}reason{R}: {L}a {L}quoted{R} word{R}}}",  # inner smart quotes
            f"{{{L}decision{R}: PASS}}",                            # unquoted value
            f"{{{L}decision{R}: {L}PASS{R},}}",                      # trailing comma
            f"{{{L}reason{R}: {L}no decision{R}}}",                  # missing decision
        ]
        for raw in still_invalid:
            with self.subTest(raw=raw):
                result = format_normalized_parse(raw)
                self.assertIsNone(result.decision)
                self.assertTrue(result.applied)
                self.assertTrue(result.error.startswith("still_invalid_after_normalization"))

    def test_other_formatting_problems_are_not_normalized(self):
        not_eligible = {
            "```json\n" + smart("PASS", "x") + "\n```": "unparseable_response",  # code fence
            "﻿" + smart("PASS", "x"): "unparseable_response",               # BOM
            f"{L}PASS{R}": "unparseable_response",                                # bare quoted label
            "{‘decision’: ‘PASS’}": None,                    # single smart quotes
            '{＂decision＂: ＂PASS＂}': None,                    # fullwidth quotes
            "": "empty_response",
            None: "empty_response",
        }
        for raw, error in not_eligible.items():
            with self.subTest(raw=raw):
                result = format_normalized_parse(raw)
                self.assertIsNone(result.decision)
                self.assertFalse(result.applied)
                if error:
                    self.assertEqual(result.error, error)
                else:
                    self.assertEqual(result.error, "invalid_json")

    def test_input_string_is_not_mutated(self):
        raw = smart("FAIL", "x")
        copy = str(raw)
        format_normalized_parse(raw)
        self.assertEqual(raw, copy)
        self.assertIn(L, raw)


if __name__ == "__main__":
    unittest.main()
