# Manual real-LLM evaluation through the Claude chat interface

This is the first **real** model experiment in TestGuard-Eval. It runs without any API calls: a
person pastes each prompt into the Claude chat interface and saves the answer exactly as given.
Results are tagged `REAL_MANUAL_LLM_EVALUATION`, which keeps them separate from the simulated
mock experiments (`SIMULATED_MOCK_NOT_REAL_LLM_PERFORMANCE`).

The code never queries a model and never writes response files. Every answer that gets scored
must have come from a real chat conversation.

## Steps

### 1. Export the prompts

```bash
python3 -m src.manual_eval export --case-ids TC001,TC004,TC011
```

This writes:
- `manual_prompts/TC001.txt`, `TC004.txt`, `TC011.txt`: paste-ready prompts. Each one contains the
  decision policy and that case's measurements only. There are no labels, reference decisions or
  case IDs inside the text.
- `manual_prompts/manifest.json`: a SHA-256 hash of each prompt, so scoring can detect if a prompt
  has changed since export.
- `manual_responses/metadata.json`: a template for you to fill in. It is only created if it
  doesn't exist yet, so your entries are never overwritten.

### 2. Prepare the chat environment (once)

Before the first case, write down your settings in `chat_settings` (step 4). Ideally:
- turn off memory and any custom or project instructions, and don't use a Project;
- turn off web search and other tools;
- use the default style;
- note whether extended thinking is on or off, and keep that setting the same for every case.

### 3. For each case, use a NEW conversation

Repeat for TC001, TC004 and TC011:

1. Open a **new chat** in Claude. Never reuse a conversation: an earlier case would leak
   into the context.
2. Note the exact model name shown in the model picker.
3. Copy the whole prompt file. On macOS:
   ```bash
   pbcopy < manual_prompts/TC001.txt
   ```
   Otherwise, open the file in a plain-text editor, select all and copy.
4. Paste it as the first message and send it. Don't add, remove or rephrase anything.
5. Don't regenerate, edit, follow up or give feedback. The **first** answer is the result.
6. Copy the answer with the chat's **copy button**, then save it unedited:
   ```bash
   pbpaste > manual_responses/TC001.txt
   ```
   Don't fix formatting, remove code fences or trim text. If you save it in an editor, save it
   as **plain UTF-8 text**: in macOS TextEdit, choose Format → Make Plain Text first, otherwise
   it saves RTF.
7. If something goes wrong (wrong model, accidental edit, interrupted reply), write it down in
   `operator_limitations`. Don't silently redo the case.

### 4. Fill in the metadata

Edit `manual_responses/metadata.json` and replace every `FILL_IN` value:

```json
{
  "model_name": "<exact model name from the model picker>",
  "experiment_date": "YYYY-MM-DD",
  "interface": "claude.ai web chat, Pro plan, manual copy/paste",
  "chat_settings": "extended thinking off; web search off; memory off; no project; default style",
  "operator_limitations": ""
}
```

Scoring refuses to run (exit code 2, nothing written) while any field is missing or still
contains `FILL_IN`, or if the date isn't in `YYYY-MM-DD` format.

### 5. Score

```bash
python3 -m src.manual_eval score --case-ids TC001,TC004,TC011
```

- Each raw answer is read byte for byte, its SHA-256 is recorded, and it is parsed with the same
  strict Stage 2 parser. A JSON object with a `decision`, or a bare label, is accepted.
- A missing file, undecodable bytes, an answer wrapped in a code fence, prose, or an unknown
  label is recorded as **INVALID**. It counts as an error and is never treated as PASS.
- The **same** parsed answers are then passed through `src/policy_gate.py`, and both arms are
  scored against the reference labels. The labels are read only at this point.
- The outputs are `results/manual_comparison.csv` (per-case answers, raw text and hashes) and
  `results/manual_summary.json` (metadata, limitations, metric definitions, both arms' metrics
  and the vetoed cases).

## Secondary analysis: FORMAT_NORMALIZED

`score` always reports **two** evaluation modes from the same raw answers:

| Mode | Role | Parsing |
|---|---|---|
| `STRICT` | **Primary.** Top-level `baseline` / `guarded` in `manual_summary.json` | The Stage 2 strict parser, unchanged |
| `FORMAT_NORMALIZED` | **Secondary.** In `format_normalized` and `mode_comparison` | One documented normalization, then the same strict parser |

The normalization rule is `typographic-double-quotes-v1` (`src/normalization.py`):

1. It is applied only when the STRICT parse failed with `invalid_json`.
2. On an **in-memory copy**, it replaces U+201C, U+201D, U+201E and U+201F (typographic double
   quotes) with U+0022 (a straight double quote). Nothing else changes: the length stays the
   same, and labels, reasons, numbers, apostrophes (e.g. U+2019), whitespace, BOMs and code fences
   are untouched. Nothing is added or removed.
3. The copy is re-parsed with the strict parser. If it is still invalid, the answer stays
   `INVALID` (`still_invalid_after_normalization:<reason>`).
4. Each case records:
   - the code points replaced, their counts and positions;
   - the SHA-256 of the original file, which is unchanged;
   - the SHA-256 of the normalized copy.

   The original file is never written to.

Report STRICT and FORMAT_NORMALIZED coverage and accuracy side by side. Never replace the
STRICT figures with the normalized ones.

## Finding from the first manual run (2026-11-09, Claude Opus 5.5, Claude Desktop chat)

**STRICT:** all 3 answers were `INVALID` (`invalid_json` at character 1), so coverage was 0/3.

**Cause:** in every file, all 8 JSON quote characters are typographic quotes. Four are U+201C LEFT
DOUBLE QUOTATION MARK (UTF-8 `e2 80 9c`), placed before each key or value, and four are
U+201D RIGHT DOUBLE QUOTATION MARK (`e2 80 9d`), placed after each. There are no straight quotes
(U+0022). TC004 also contains U+2019 RIGHT SINGLE QUOTATION MARK in "note’s". That is ordinary
text inside the reason and is not a parsing problem.

**FORMAT_NORMALIZED:** all 3 answers parse, coverage is 3/3, and the decisions are TC001 PASS,
TC004 INSUFFICIENT_DATA and TC011 FAIL. All match the reference labels, so the gate vetoed
nothing.

**Origin: cannot be determined from the saved files.** The evidence fits two explanations equally
well:
- **The model typeset the quotes itself.** The prompt shows straight quotes, and models
  usually emit straight quotes in JSON, but this is not guaranteed.
- **The quotes were converted after the answer was produced.** Possible causes are a "smart
  quotes" feature in a text editor or another app on the copy-and-save path, or the chat app's
  copy behaviour. The placement (left quote after `{` or a space, right quote after a letter or
  period) is exactly what macOS-style smart-quote substitution produces. The files have no
  trailing newline, which is consistent with both `pbpaste` and many editors.

The metadata doesn't record how the files were saved, so the cause cannot be settled after the
fact.

**How to decide it:**
1. Open one of the original conversations and click the answer's copy button.
2. Without pasting it anywhere else, run:

   ```bash
   pbpaste | xxd | head -2
   ```

3. Read the bytes right after `7b` (`{`):
   - `22` means the answer is straight-quoted at source, so the quotes were changed while saving
     (a copy/paste or editor artifact).
   - `e2 80 9c` means the typographic quotes came from the model output (or from the copy button
     itself).

Record what you find in `operator_limitations`. Don't overwrite the original response files.

**What this means for the analysis:** in STRICT mode, the first run measures format compliance
across the whole collection pipeline, not decision quality. The FORMAT_NORMALIZED decisions are
correct on 3/3 cases, but with n = 3 and a single sample per case, this is an anecdote and not a
performance estimate.

**Change to the protocol for future runs:** save answers only with
`pbpaste > manual_responses/<ID>.txt`, never through a rich-text editor. Record the save method in
`chat_settings` or `operator_limitations`, and check each file with `xxd` before scoring.

## Methodology limitations

- **Manual UI.** Prompts and answers are copied by hand, so transcription errors are possible.
  The hashes only detect changes made after saving.
- **Unknown sampling parameters.** Temperature, top_p, maximum length and the chat product's own
  system prompt are not controlled and not recorded.
- **No true system-role control.** In the API, the policy would be sent as a system prompt. Here
  it is pasted as part of the user message, where it competes with the product's built-in system
  prompt. Results may differ from an API run with the same text.
- **Product features.** Memory, custom instructions, styles, extended thinking and tools can all
  change answers, so record them in `chat_settings`.
- **Single sample per case.** With no repeated runs, variance is unknown.
- **Small sample size.** With 3 cases, each rate moves in steps of 33–100 percentage points, so no
  statistical claim is possible.
- **The gate shares the policy with the labels.** Any improvement of the guarded arm is expected
  by construction.
- **Normalization is a secondary lens.** FORMAT_NORMALIZED fixes only typographic double quotes. It
  cannot tell a model formatting error apart from a copy/paste artifact, and it must not be used
  to hide format failures.
- **Strict format scoring.** An answer that is correct but formatted wrongly (e.g. wrapped in a
  ```` ```json ```` fence) is counted as INVALID. This measures whether the model follows the
  format, not only whether its decision is right.

## Privacy and version control

`manual_prompts/` and `manual_responses/` are in `.gitignore`. Raw transcripts are working data.
If you later choose anonymized examples to publish, copy them to a dedicated folder (for example
`examples/manual/`) and commit them deliberately.
