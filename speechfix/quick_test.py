# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
quick_test.py -- Command-line smoke test (no Streamlit needed)
==============================================================
Run this to verify the full pipeline works before launching the UI.

Usage:
    d:\\project\\.venv\\Scripts\\python.exe d:\\project\\speechfix\\quick_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import data_loader, asr as asr_module, grammar as grammar_module, evaluation as eval_module


DATASET_ROOT = r"d:\project\LJSpeech-1.1"
N_SAMPLES    = 3
MODEL_SIZE   = "base"


def divider(char="-", width=70):
    print(char * width)


def run():
    print()
    divider("=")
    print("  [SPEECHFIX] Quick Test -- Full Pipeline")
    divider("=")

    print(f"\n[A] Loading metadata from: {DATASET_ROOT}")
    records   = data_loader.load_metadata(DATASET_ROOT, max_samples=N_SAMPLES)
    available = [r for r in records if r["exists"]]

    if not available:
        print("ERROR: No .wav files found! Check DATASET_ROOT path.")
        sys.exit(1)

    print(f"   Found {len(available)} clips.\n")

    aggregate = {"asr_wer": [], "corr_wer": [], "delta": []}

    for idx, rec in enumerate(available, 1):
        divider()
        print(f"  Clip {idx}/{len(available)}: {rec['id']}")
        divider()
        print(f"  [REF] Ground Truth : {rec['ground_truth']}")

        print(f"\n  [A] Block A -- Whisper ({MODEL_SIZE}) transcribing...")
        asr_r = asr_module.transcribe(rec["wav_path"], MODEL_SIZE)
        print(f"      ASR text : {asr_r['text']}")
        print(f"      Time     : {asr_r['duration_s']}s  |  Device: {asr_r['device'].upper()}")

        print(f"\n  [B] Block B -- LanguageTool grammar correction...")
        gram_r = grammar_module.correct(asr_r["text"])
        print(f"      Corrected   : {gram_r['corrected_text']}")
        print(f"      Corrections : {gram_r['num_corrections']}  "
              f"(rules: {', '.join(gram_r['rules_triggered']) or 'none'})")

        print(f"\n  [C] Block C -- WER Evaluation...")
        ev_r = eval_module.compare(rec["ground_truth"], asr_r["text"], gram_r["corrected_text"])
        print(f"      WER  (ASR)       : {ev_r['asr_metrics']['wer_pct']}")
        print(f"      WER  (Corrected) : {ev_r['corrected_metrics']['wer_pct']}")
        print(f"      dWER             : {ev_r['delta_wer']:+.4f}")
        print(f"      % Improvement    : {ev_r['improvement_pct']:+.2f}%")
        print(f"\n  {ev_r['verdict']}")

        print(f"\n  [MATH] Mathematical Proof:")
        asr_m  = ev_r["asr_metrics"]
        corr_m = ev_r["corrected_metrics"]
        print(f"      WER_asr  = ({asr_m['substitutions']}+{asr_m['deletions']}+{asr_m['insertions']}) / {asr_m['ref_words']} = {asr_m['wer']:.4f}")
        print(f"      WER_corr = ({corr_m['substitutions']}+{corr_m['deletions']}+{corr_m['insertions']}) / {corr_m['ref_words']} = {corr_m['wer']:.4f}")
        print(f"      dWER     = {asr_m['wer']:.4f} - {corr_m['wer']:.4f} = {ev_r['delta_wer']:.4f}")
        print(f"      Gain     = (dWER / WER_asr) x 100 = {ev_r['improvement_pct']:.2f}%")

        aggregate["asr_wer"].append(asr_m["wer"])
        aggregate["corr_wer"].append(corr_m["wer"])
        aggregate["delta"].append(ev_r["delta_wer"])
        print()

    if len(available) > 1:
        divider("=")
        print("  [SUMMARY] AGGREGATE SUMMARY")
        divider("=")
        avg_asr   = sum(aggregate["asr_wer"])  / len(aggregate["asr_wer"])
        avg_corr  = sum(aggregate["corr_wer"]) / len(aggregate["corr_wer"])
        avg_delta = sum(aggregate["delta"])     / len(aggregate["delta"])
        avg_gain  = (avg_delta / avg_asr * 100) if avg_asr > 0 else 0
        print(f"  Avg WER  (ASR)       : {avg_asr*100:.2f}%")
        print(f"  Avg WER  (Corrected) : {avg_corr*100:.2f}%")
        print(f"  Avg dWER             : {avg_delta:+.4f}")
        print(f"  Avg % Improvement    : {avg_gain:+.2f}%")
        print()
        if avg_delta > 0:
            print("  [OK] Grammar correction improved accuracy on average.")
        elif avg_delta < 0:
            print("  [WARN] Grammar correction slightly degraded accuracy on average.")
        else:
            print("  [INFO] Grammar correction had no net effect.")

    divider("=")
    print("  All done! Launch the Streamlit UI with:")
    print("     d:\\project\\speechfix\\run.bat")
    divider("=")
    print()


if __name__ == "__main__":
    run()
