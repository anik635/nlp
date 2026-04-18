"""
Block B: The Brain (Grammar Correction Engine)
===============================================
Uses LanguageTool (via language_tool_python) to detect and fix grammatical,
punctuation, spelling, and style errors in the ASR transcript.

How LanguageTool works:
-----------------------
LanguageTool is a rule-based NLP engine with 4,000+ hand-crafted linguistic
rules for English. When you pass text through it:

  1. Tokenisation -- splits text into words and sentences.
  2. POS Tagging  -- labels each token (noun, verb, adjective, etc.).
  3. Rule Matching -- applies patterns like:
       - "a" before a vowel -> "an"
       - double spaces -> single space
       - subject-verb disagreement (e.g. "he go" -> "he goes")
  4. Replacement   -- swaps the flagged token with the best suggestion.
"""

import language_tool_python
import difflib
import time

_tool_cache: dict = {}


def _get_tool(language: str = "en-US") -> language_tool_python.LanguageTool:
    """Return a cached LanguageTool instance."""
    if language not in _tool_cache:
        _tool_cache[language] = language_tool_python.LanguageTool(language)
    return _tool_cache[language]


def correct(text: str, language: str = "en-US") -> dict:
    """
    Run grammar correction on text and return rich diagnostics.

    Returns
    -------
    dict with keys:
        corrected_text  - the fixed string
        matches         - list of LanguageTool match objects
        num_corrections - integer count of changes applied
        duration_s      - processing time in seconds
        rules_triggered - list of rule IDs that fired (e.g. "EN_A_VS_AN")
    """
    tool = _get_tool(language)

    t0      = time.perf_counter()
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    elapsed = time.perf_counter() - t0

    rules = [m.rule_id for m in matches]

    return {
        "corrected_text":  corrected.strip(),
        "matches":         matches,
        "num_corrections": len(matches),
        "duration_s":      round(elapsed, 2),
        "rules_triggered": rules,
    }


def build_diff_html(original: str, corrected: str) -> str:
    """
    Generate word-level HTML diff between original and corrected text.

    Returns an HTML string where:
      - deleted words are wrapped in <span class="diff-del">...</span>
      - inserted words are wrapped in <span class="diff-ins">...</span>
      - unchanged words are plain text
    """
    orig_words = original.split()
    corr_words = corrected.split()

    sm    = difflib.SequenceMatcher(None, orig_words, corr_words, autojunk=False)
    parts = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(orig_words[i1:i2]))
        elif tag == "replace":
            parts.append(
                f'<span class="diff-del">{" ".join(orig_words[i1:i2])}</span> '
                f'<span class="diff-ins">{" ".join(corr_words[j1:j2])}</span>'
            )
        elif tag == "delete":
            parts.append(f'<span class="diff-del">{" ".join(orig_words[i1:i2])}</span>')
        elif tag == "insert":
            parts.append(f'<span class="diff-ins">{" ".join(corr_words[j1:j2])}</span>')

    return " ".join(parts)


def summarise_errors(matches) -> list:
    """
    Convert raw LanguageTool matches into human-readable summaries.

    Returns a list of dicts:
        rule_id   - machine identifier (e.g. "EN_A_VS_AN")
        message   - plain-English description of the error
        context   - the exact phrase in the original text
        suggestion - the recommended replacement
        category  - linguistic category (GRAMMAR, TYPOS, STYLE, ...)
    """
    summaries = []
    for m in matches:
        summaries.append({
            "rule_id":    m.rule_id,
            "message":    m.message,
            "context":    m.context,
            "suggestion": m.replacements[0] if m.replacements else "n/a",
            "category":   m.category,
        })
    return summaries
