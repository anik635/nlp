# SpeechFix

A local pipeline that takes a `.wav` file, figures out what was said, fixes any grammar mistakes the transcription made, and then shows you mathematically whether the fix actually helped.

No cloud APIs. No subscriptions. Runs entirely on your machine.

---

## What it does

Speech recognition is good, but not perfect. Whisper might transcribe "and graved in relief" when the speaker actually said "engraved in relief". Or it might miss capitalisation, mess up punctuation, or confuse homophones.

SpeechFix runs that raw transcript through LanguageTool — a grammar engine with 4,000+ linguistic rules — and then calculates the Word Error Rate before and after the correction to show whether the fix made things better or worse. The math is shown step by step so you can verify it yourself.

**The four stages:**

1. **Whisper** listens to the audio and produces a raw transcript
2. **LanguageTool** scans that transcript and fixes grammar, spelling, and punctuation
3. **WER evaluation** compares both versions against the ground truth and computes the improvement
4. **Streamlit dashboard** shows you all of this in a clean UI

---

## Getting started

You need Python 3.9+ and the virtual environment already set up. If you're cloning this fresh:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r speechfix/requirements.txt
```

The first time you run the app, two things will download automatically:
- **Whisper base model** (~139 MB) — cached after the first run
- **LanguageTool** (~259 MB) — also cached, never downloads again

Then just launch:

```bash
speechfix\run.bat
```

Or if you prefer:

```bash
.venv\Scripts\streamlit run speechfix/app.py
```

Open `http://localhost:8501` in your browser.

---

## Using the app

**Upload mode** is the main way to use it. Drop any `.wav` file into the uploader and hit Run. If you have the original text that was spoken, paste it in the reference box — that's what unlocks the WER chart and the mathematical proof. If you don't have a reference, you'll still get the transcription and grammar correction; you just won't get WER numbers.

**Dataset mode** works if you have the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/) downloaded. It's 13,100 clips of a single speaker reading passages from public domain books, and it comes with ground-truth transcriptions — perfect for seeing WER improvement in action. Set the path in the sidebar and pick any clip from the dropdown.

**Batch mode** lets you run the pipeline across multiple clips at once and get an aggregate comparison table and chart.

---

## Project structure

```
speechfix/
├── app.py                  main Streamlit app
├── quick_test.py           run this first to verify everything works without a browser
├── run.bat                 Windows launcher
├── requirements.txt
└── pipeline/
    ├── asr.py              Block A — Whisper transcription
    ├── grammar.py          Block B — LanguageTool correction + diff highlighting
    ├── evaluation.py       Block C — WER calculation and math proof
    └── data_loader.py      loads LJ Speech metadata.csv
```

---

## The math behind it

Word Error Rate is defined as:

```
WER = (Substitutions + Deletions + Insertions) / Number of reference words
```

It's computed using the Levenshtein edit distance algorithm at the word level. A WER of 0.0 means the transcript matched perfectly. A WER of 0.2 means roughly 1 in 5 words was wrong.

After grammar correction, we compute:

```
ΔWER = WER_before - WER_after
```

If `ΔWER > 0`, the corrected text was closer to what was actually said. The dashboard shows the full breakdown — substitutions, deletions, insertions — for both the raw and corrected transcripts side by side.

---

## Things worth knowing

- **No ffmpeg needed.** The audio loading uses Python's built-in `wave` module, so there's nothing extra to install.
- **LJ Speech is optional.** The upload tab works completely standalone — just drag in a `.wav` file.
- **Model size tradeoff.** `tiny` is the fastest (2–3 seconds per clip) but misses more words. `small` is more accurate but takes around 10 seconds. `base` is the default and hits a reasonable middle ground.
- **LanguageTool is rule-based**, not an LLM, so it won't hallucinate corrections. It only changes things it's confident about.
- **WER can go up.** If LanguageTool changes a word that was actually correct in the transcript, WER gets worse. This is expected and the dashboard will flag it.

---

## Running the CLI test

If you want to verify the pipeline works before opening the browser:

```bash
.venv\Scripts\python speechfix/quick_test.py
```

This runs 3 clips from the LJ Speech dataset through all three blocks and prints the results with the full math proof inline.

---

## Dataset

[LJ Speech](https://keithito.com/LJ-Speech-Dataset/) is a public domain dataset of 13,100 short audio clips (about 24 hours total) recorded by a single speaker. The transcriptions are clean and human-verified, which makes it reliable for WER evaluation. The dataset is not included in this repo — download it separately and point the sidebar to wherever you extracted it.

---

## Dependencies

| Package | What it does |
|---|---|
| `openai-whisper` | Speech-to-text model |
| `language-tool-python` | Grammar checker (wraps LanguageTool's Java engine) |
| `jiwer` | WER / edit distance calculation |
| `streamlit` | Web UI |
| `matplotlib` | WER comparison charts |
| `torch` | Required by Whisper for model inference |

---

Built as a demonstration of how ASR post-processing can be evaluated rigorously rather than just eyeballed.
