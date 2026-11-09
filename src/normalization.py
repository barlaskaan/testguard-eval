"""Secondary FORMAT_NORMALIZED parsing: one minimal, documented transformation.

The primary result is always the STRICT parse (``llm_baseline.parse_response``).
This module adds a secondary analysis for one known failure mode: JSON whose
double quotes are typographic ("smart") quotes instead of U+0022.

Rule (NORMALIZATION_VERSION = "typographic-double-quotes-v1"):
  * Eligible only if the STRICT parse failed with ``invalid_json``, i.e. the
    response looks like a JSON object but json.loads rejected it.
  * On an in-memory copy, replace each of these code points with U+0022 QUOTATION MARK:
      U+201C LEFT DOUBLE QUOTATION MARK
      U+201D RIGHT DOUBLE QUOTATION MARK
      U+201E DOUBLE LOW-9 QUOTATION MARK
      U+201F DOUBLE HIGH-REVERSED-9 QUOTATION MARK
  * Nothing else is changed: no single quotes/apostrophes (e.g. U+2019), no
    whitespace, BOMs, code fences, braces, labels, reasons or numbers. Nothing
    is inserted or deleted, so the text keeps the same length and every other
    character is identical.
  * The copy is re-parsed with the same strict parser. If it is still invalid,
    the result stays INVALID.
The original text and file are never modified.
"""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field

from .llm_baseline import parse_response

FORMAT_NORMALIZED = "FORMAT_NORMALIZED"
STRICT = "STRICT"
NORMALIZATION_VERSION = "typographic-double-quotes-v1"
TYPOGRAPHIC_DOUBLE_QUOTES = ("“", "”", "„", "‟")
STRAIGHT_QUOTE = '"'
NORMALIZATION_RULES = [
    "Applied only when the STRICT parse failed with invalid_json.",
    "Replace U+201C, U+201D, U+201E and U+201F with U+0022 on an in-memory copy; change nothing else.",
    "Re-parse with the same strict parser; if still invalid, the response stays INVALID.",
    "Single quotes/apostrophes (e.g. U+2019), BOMs, code fences, whitespace and content are never changed.",
]


def _codepoint(ch: str) -> str:
    return f"U+{ord(ch):04X}"


@dataclass(frozen=True)
class NormalizedParse:
    decision: str | None              # None => INVALID in FORMAT_NORMALIZED mode
    error: str | None
    eligible: bool                    # strict failure was invalid_json
    applied: bool                     # at least one character was replaced
    transformations: tuple[dict, ...] = field(default_factory=tuple)
    normalized_sha256: str | None = None


def normalize_typographic_double_quotes(text: str) -> tuple[str, tuple[dict, ...]]:
    """Return (normalized_copy, transformations). Length is always preserved."""
    positions: dict[str, list[int]] = {}
    chars = []
    for index, ch in enumerate(text):
        if ch in TYPOGRAPHIC_DOUBLE_QUOTES:
            positions.setdefault(ch, []).append(index)
            chars.append(STRAIGHT_QUOTE)
        else:
            chars.append(ch)
    transformations = tuple(
        {
            "rule": NORMALIZATION_VERSION,
            "from": _codepoint(ch),
            "from_name": unicodedata.name(ch),
            "to": _codepoint(STRAIGHT_QUOTE),
            "count": len(idx),
            "positions": idx,
        }
        for ch, idx in sorted(positions.items())
    )
    return "".join(chars), transformations


def format_normalized_parse(raw: str | None) -> NormalizedParse:
    """Parse ``raw`` in FORMAT_NORMALIZED mode (secondary analysis)."""
    strict = parse_response(raw)
    if strict.decision is not None:
        return NormalizedParse(strict.decision, None, eligible=False, applied=False)
    if strict.error != "invalid_json" or not isinstance(raw, str):
        return NormalizedParse(None, strict.error, eligible=False, applied=False)

    normalized, transformations = normalize_typographic_double_quotes(raw)
    if not transformations:
        return NormalizedParse(None, "invalid_json", eligible=True, applied=False)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    reparsed = parse_response(normalized)
    if reparsed.decision is None:
        return NormalizedParse(None, f"still_invalid_after_normalization:{reparsed.error}",
                               eligible=True, applied=True, transformations=transformations,
                               normalized_sha256=digest)
    return NormalizedParse(reparsed.decision, None, eligible=True, applied=True,
                           transformations=transformations, normalized_sha256=digest)
