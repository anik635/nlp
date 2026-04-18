"""
Block C: The Judge (WER Evaluation & Mathematical Proof)
=========================================================
Calculates Word Error Rate (WER) using the jiwer library.

Mathematical definition of WER:

        S  +  D  +  I
  WER = -----------------
               N

  Where:
    S = Substitutions  (wrong word used)
    D = Deletions      (word missing from hypothesis)
    I = Insertions     (extra word added by ASR)
    N = Total words in the Reference (ground truth)

  WER = 0.0  <- perfect transcript
  WER = 1.0  <- every word is wrong
  WER > 1.0  <- more errors than words (too many insertions)

How the edit-distance is computed (Levenshtein algorithm):
----------------------------------------------------------
WER is the word-level Levenshtein edit distance normalised by the reference
length. jiwer uses a dynamic-programming table:

  - Build an (M+1) x (N+1) matrix where M = len(hypothesis words)
    and N = len(reference words).
  - dp[i][j] = minimum edits to transform hyp[:i] into ref[:j].
  - Fill left-to-right, top-to-bottom with substitution/deletion/
    insertion costs (all = 1).
  - The bottom-right cell dp[M][N] = total edit distance.
"""

from jiwer import process_words
import re


def _normalise(text: str) -> str:
    """
    Lowercase and strip punctuation before WER calculation.
    Matches the convention used in standard ASR evaluation benchmarks.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_wer(reference: str, hypothesis: str) -> dict:
    """
    Compute WER and retrieve the underlying edit operations.

    Parameters
    ----------
    reference  : str   Ground-truth transcription.
    hypothesis : str   Text produced by ASR or grammar corrector.

    Returns
    -------
    dict with keys:
        wer           - float  (0.0 = perfect, >0 = errors exist)
        wer_pct       - WER as a percentage string, e.g. "12.5%"
        substitutions, deletions, insertions - raw counts
        ref_words     - total words in the reference
    """
    ref_clean = _normalise(reference)
    hyp_clean = _normalise(hypothesis)

    output = process_words(ref_clean, hyp_clean)
    score  = output.wer

    return {
        "wer":           round(score, 4),
        "wer_pct":       f"{score * 100:.2f}%",
        "substitutions": output.substitutions,
        "deletions":     output.deletions,
        "insertions":    output.insertions,
        "ref_words":     len(ref_clean.split()),
    }


def compare(reference: str, asr_text: str, corrected_text: str) -> dict:
    """
    Run a full three-way comparison and compute the improvement delta.

    Returns
    -------
    dict with keys:
        asr_metrics       - calculate_wer result for raw ASR text
        corrected_metrics - calculate_wer result for grammar-corrected text
        delta_wer         - WER_asr - WER_corrected  (positive = improvement)
        improvement_pct   - percentage improvement over the ASR baseline
        verdict           - human-readable conclusion string
        math_proof        - formatted proof string for display
    """
    asr_m  = calculate_wer(reference, asr_text)
    corr_m = calculate_wer(reference, corrected_text)

    delta   = round(asr_m["wer"] - corr_m["wer"], 4)
    imp_pct = (delta / asr_m["wer"] * 100) if asr_m["wer"] > 0 else 0.0

    if delta > 0.001:
        verdict = f"Grammar correction IMPROVED accuracy by {imp_pct:.1f}%"
    elif delta < -0.001:
        verdict = f"Grammar correction degraded accuracy by {abs(imp_pct):.1f}%"
    else:
        verdict = "Grammar correction had no measurable effect on WER"

    math_proof = (
        f"**Mathematical Proof of Improvement**\n\n"
        f"```\n"
        f"Reference words (N)   = {asr_m['ref_words']}\n\n"
        f"--- ASR Transcript ---\n"
        f"  Substitutions (S)   = {asr_m['substitutions']}\n"
        f"  Deletions     (D)   = {asr_m['deletions']}\n"
        f"  Insertions    (I)   = {asr_m['insertions']}\n"
        f"  WER_asr             = (S+D+I)/N = "
        f"({asr_m['substitutions']}+{asr_m['deletions']}+{asr_m['insertions']})/{asr_m['ref_words']} "
        f"= {asr_m['wer']:.4f}  ({asr_m['wer_pct']})\n\n"
        f"--- Corrected Transcript ---\n"
        f"  Substitutions (S)   = {corr_m['substitutions']}\n"
        f"  Deletions     (D)   = {corr_m['deletions']}\n"
        f"  Insertions    (I)   = {corr_m['insertions']}\n"
        f"  WER_corrected       = (S+D+I)/N = "
        f"({corr_m['substitutions']}+{corr_m['deletions']}+{corr_m['insertions']})/{corr_m['ref_words']} "
        f"= {corr_m['wer']:.4f}  ({corr_m['wer_pct']})\n\n"
        f"--- Improvement ---\n"
        f"  ΔWER  = WER_asr - WER_corrected\n"
        f"        = {asr_m['wer']:.4f} - {corr_m['wer']:.4f}\n"
        f"        = {delta:.4f}\n\n"
        f"  % Improvement = (ΔWER / WER_asr) x 100\n"
        f"                = ({delta:.4f} / {asr_m['wer']:.4f}) x 100\n"
        f"                = {imp_pct:.2f}%\n"
        f"```"
    )

    return {
        "asr_metrics":       asr_m,
        "corrected_metrics": corr_m,
        "delta_wer":         delta,
        "improvement_pct":   round(imp_pct, 2),
        "verdict":           verdict,
        "math_proof":        math_proof,
    }
