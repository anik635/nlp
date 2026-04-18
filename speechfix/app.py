"""
SpeechFix -- Main Streamlit Application (Neo-Brutalist UI)
===========================================================
Block D: The Face

Design: Neo-brutalism with organic softening
  - Thick black borders, offset box-shadows
  - Warm cream background
  - Smooth CSS transitions + 3D perspective tilts on cards
  - Space Grotesk typography
  - Dataset is FULLY OPTIONAL -- upload any .wav directly
"""

import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import sys
import time
from pathlib import Path

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import asr as asr_module
from pipeline import grammar as grammar_module
from pipeline import evaluation as eval_module
from pipeline import data_loader


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SpeechFix",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
  --cream:       #F5F0E8;
  --cream-dark:  #EDE8DC;
  --ink:         #0D0D0D;
  --ink-soft:    #2A2A2A;
  --border:      2.5px solid #0D0D0D;
  --shadow:      5px 5px 0px #0D0D0D;
  --shadow-hover:8px 8px 0px #0D0D0D;
  --shadow-xl:   10px 10px 0px #0D0D0D;
  --r:           6px;
  --yellow:      #FFD93D;
  --orange:      #FF6B35;
  --blue:        #4361EE;
  --green:       #06D6A0;
  --red:         #EF233C;
  --trans:       all 0.22s cubic-bezier(0.34,1.56,0.64,1);
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif !important;
  background-color: var(--cream) !important;
  color: var(--ink) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; }

section[data-testid="stSidebar"] {
  background: var(--cream-dark) !important;
  border-right: var(--border);
}
section[data-testid="stSidebar"] .block-container {
  padding: 1.2rem 1rem !important;
}

.stTextInput > div > div,
.stSelectbox > div > div {
  border: var(--border) !important;
  border-radius: var(--r) !important;
  background: white !important;
  box-shadow: 3px 3px 0 var(--ink) !important;
  transition: var(--trans) !important;
}
.stTextInput > div > div:focus-within,
.stSelectbox > div > div:hover {
  box-shadow: 5px 5px 0 var(--ink) !important;
  transform: translate(-1px,-1px);
}

.stButton > button {
  background: var(--ink) !important;
  color: white !important;
  border: var(--border) !important;
  border-radius: var(--r) !important;
  box-shadow: var(--shadow) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 0.55rem 1.2rem !important;
  transition: var(--trans) !important;
}
.stButton > button:hover {
  transform: translate(-3px, -3px) !important;
  box-shadow: var(--shadow-hover) !important;
}
.stButton > button:active {
  transform: translate(2px, 2px) !important;
  box-shadow: 2px 2px 0 var(--ink) !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  background: transparent;
  border-bottom: var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: var(--cream-dark);
  border: var(--border);
  border-radius: var(--r) var(--r) 0 0 !important;
  padding: 0.5rem 1.2rem;
  font-weight: 700;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--ink-soft) !important;
  transition: var(--trans);
}
.stTabs [data-baseweb="tab"]:hover {
  transform: translateY(-2px);
  background: white;
}
.stTabs [aria-selected="true"] {
  background: var(--ink) !important;
  color: white !important;
  transform: translateY(-3px);
}

.sf-hero {
  border: var(--border);
  border-radius: var(--r);
  background: var(--ink);
  color: white;
  padding: 2rem 2.5rem 1.8rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
  box-shadow: var(--shadow-xl);
}
.sf-hero::before {
  content: '';
  position: absolute;
  top: -40%; right: -15%;
  width: 420px; height: 420px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,217,61,0.18) 0%, transparent 65%);
  pointer-events: none;
}
.sf-hero::after {
  content: '';
  position: absolute;
  bottom: -30%; left: 30%;
  width: 300px; height: 300px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(67,97,238,0.15) 0%, transparent 65%);
  pointer-events: none;
}
.sf-hero h1 {
  font-size: 2.6rem;
  font-weight: 700;
  margin: 0 0 0.3rem;
  line-height: 1.1;
  letter-spacing: -0.02em;
}
.sf-hero h1 span { color: var(--yellow); }
.sf-hero p { color: rgba(255,255,255,0.65); margin: 0; font-size: 1rem; }
.pipeline-row { display: flex; gap: 0.5rem; margin-top: 1.2rem; flex-wrap: wrap; }
.p-badge {
  padding: 0.3rem 0.8rem;
  border-radius: 99px;
  font-size: 0.77rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  border: 1.5px solid rgba(255,255,255,0.2);
}
.p-badge-a { background: rgba(67,97,238,0.3);  color: #a5b4fc; }
.p-badge-b { background: rgba(6,214,160,0.3);  color: #6ee7b7; }
.p-badge-c { background: rgba(255,217,61,0.3); color: #fde68a; }
.p-badge-d { background: rgba(255,107,53,0.3); color: #fca5a5; }

.sf-card {
  background: white;
  border: var(--border);
  border-radius: var(--r);
  padding: 1.2rem 1.4rem;
  margin-bottom: 1rem;
  box-shadow: var(--shadow);
  transition: var(--trans);
  transform-style: preserve-3d;
  perspective: 800px;
}
.sf-card:hover {
  transform: translate(-3px, -3px) rotateX(1.5deg) rotateY(-1deg);
  box-shadow: var(--shadow-hover);
}
.sf-card h3 {
  margin: 0 0 0.4rem;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: rgba(0,0,0,0.45);
}
.sf-card .val {
  font-size: 2.1rem;
  font-weight: 700;
  line-height: 1;
  font-family: 'Space Mono', monospace;
}
.sf-card .sub {
  font-size: 0.75rem;
  color: rgba(0,0,0,0.4);
  margin-top: 0.35rem;
  font-family: 'Space Mono', monospace;
}

.sf-card-yellow { border-left: 6px solid var(--yellow) !important; }
.sf-card-blue   { border-left: 6px solid var(--blue)   !important; }
.sf-card-green  { border-left: 6px solid var(--green)  !important; }
.sf-card-orange { border-left: 6px solid var(--orange) !important; }
.sf-card-red    { border-left: 6px solid var(--red)    !important; }

.sf-section {
  display: flex; align-items: center; gap: 0.6rem;
  font-size: 0.82rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--ink-soft);
  margin: 1.4rem 0 0.7rem;
  padding-bottom: 0.4rem;
  border-bottom: 2px solid var(--ink);
}

.tx-box {
  background: white;
  border: var(--border);
  border-radius: var(--r);
  padding: 1.1rem 1.3rem;
  font-size: 0.95rem;
  line-height: 1.75;
  min-height: 70px;
  box-shadow: 4px 4px 0 var(--ink);
  transition: var(--trans);
}
.tx-box:hover {
  transform: translate(-2px, -2px);
  box-shadow: 6px 6px 0 var(--ink);
}
.diff-del {
  background: rgba(239,35,60,0.12);
  color: #c00;
  text-decoration: line-through;
  border-radius: 3px;
  padding: 1px 4px;
}
.diff-ins {
  background: rgba(6,214,160,0.18);
  color: #047a5a;
  font-weight: 700;
  border-radius: 3px;
  padding: 1px 4px;
}

.stFileUploader > div {
  border: 2.5px dashed var(--ink) !important;
  border-radius: var(--r) !important;
  background: white !important;
  box-shadow: 4px 4px 0 var(--ink) !important;
  transition: var(--trans) !important;
}
.stFileUploader > div:hover {
  transform: translate(-2px, -2px) !important;
  box-shadow: 6px 6px 0 var(--ink) !important;
  background: var(--cream) !important;
}

.sf-callout {
  border: var(--border);
  border-left: 5px solid var(--blue);
  border-radius: var(--r);
  background: white;
  padding: 0.75rem 1rem;
  margin: 0.8rem 0;
  font-size: 0.88rem;
  color: var(--ink-soft);
  box-shadow: 3px 3px 0 var(--ink);
}

.wer-track {
  height: 10px;
  background: var(--cream-dark);
  border: 1.5px solid var(--ink);
  border-radius: 99px;
  overflow: hidden;
  margin-top: 0.4rem;
}
.wer-fill {
  height: 100%;
  border-radius: 99px;
  transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1);
}

.err-row {
  background: white;
  border: var(--border);
  border-radius: var(--r);
  padding: 0.65rem 1rem;
  margin-bottom: 0.5rem;
  font-size: 0.84rem;
  box-shadow: 3px 3px 0 var(--ink);
  transition: var(--trans);
}
.err-row:hover {
  transform: translate(-2px, -2px);
  box-shadow: 5px 5px 0 var(--ink);
}
.rule-chip {
  font-family: 'Space Mono', monospace;
  font-size: 0.72rem;
  background: var(--yellow);
  color: var(--ink);
  border: 1px solid var(--ink);
  padding: 1px 7px;
  border-radius: 4px;
  font-weight: 700;
}

.sb-label {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--ink-soft);
  margin: 1rem 0 0.3rem;
  padding-bottom: 0.25rem;
  border-bottom: 2px solid var(--ink);
}

hr { border: none; border-top: 2px solid var(--ink); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_dataset(root: str, n: int):
    return data_loader.load_metadata(root, max_samples=n)


def run_pipeline(audio_path: str, model_size: str, reference_text: str = ""):
    """Run all 3 blocks and return results."""

    with st.status("🔵 Block A — Whisper transcribing audio…", expanded=True) as s:
        st.write(f"Loading `whisper-{model_size}`…")
        asr_r = asr_module.transcribe(audio_path, model_size)
        st.write(f"Done in **{asr_r['duration_s']}s** on `{asr_r['device'].upper()}`")
        s.update(label="🔵 Block A — ASR Complete", state="complete")

    with st.status("🟢 Block B — Grammar correction…", expanded=True) as s:
        st.write("Scanning with LanguageTool rules…")
        gram_r = grammar_module.correct(asr_r["text"])
        st.write(f"{gram_r['num_corrections']} issue(s) fixed in {gram_r['duration_s']}s")
        s.update(label=f"🟢 Block B — Grammar Complete ({gram_r['num_corrections']} fixes)", state="complete")

    eval_r = None
    if reference_text.strip():
        with st.status("🟡 Block C — WER evaluation…", expanded=True) as s:
            eval_r = eval_module.compare(reference_text, asr_r["text"], gram_r["corrected_text"])
            delta  = eval_r["delta_wer"]
            s.update(label=f"🟡 Block C — WER Complete | dWER = {delta:+.4f}", state="complete")

    return asr_r, gram_r, eval_r


def render_results(asr_r, gram_r, eval_r, reference_text=""):
    """Render the full results section."""

    st.markdown('<div class="sf-section">📊 Results</div>', unsafe_allow_html=True)

    if eval_r:
        asr_wer  = eval_r["asr_metrics"]["wer"]
        corr_wer = eval_r["corrected_metrics"]["wer"]
        delta    = eval_r["delta_wer"]
        imp_pct  = eval_r["improvement_pct"]

        c1, c2, c3, c4 = st.columns(4)
        wer_color  = "#EF233C" if asr_wer  > 0.3 else "#FF6B35" if asr_wer  > 0.1 else "#06D6A0"
        wer2_color = "#EF233C" if corr_wer > 0.3 else "#FF6B35" if corr_wer > 0.1 else "#06D6A0"
        d_color    = "#06D6A0" if delta > 0 else "#EF233C" if delta < 0 else "#888"
        imp_color  = "#06D6A0" if imp_pct > 0 else "#EF233C" if imp_pct < 0 else "#888"
        arrow      = "▼" if delta > 0 else "▲" if delta < 0 else "="

        with c1:
            st.markdown(f"""
            <div class="sf-card sf-card-blue">
              <h3>🔵 ASR WER</h3>
              <div class="val" style="color:{wer_color}">{eval_r['asr_metrics']['wer_pct']}</div>
              <div class="wer-track"><div class="wer-fill" style="width:{min(asr_wer*100,100):.1f}%;background:{wer_color}"></div></div>
              <div class="sub">S:{eval_r['asr_metrics']['substitutions']} D:{eval_r['asr_metrics']['deletions']} I:{eval_r['asr_metrics']['insertions']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="sf-card sf-card-green">
              <h3>🟢 Corrected WER</h3>
              <div class="val" style="color:{wer2_color}">{eval_r['corrected_metrics']['wer_pct']}</div>
              <div class="wer-track"><div class="wer-fill" style="width:{min(corr_wer*100,100):.1f}%;background:{wer2_color}"></div></div>
              <div class="sub">S:{eval_r['corrected_metrics']['substitutions']} D:{eval_r['corrected_metrics']['deletions']} I:{eval_r['corrected_metrics']['insertions']}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="sf-card sf-card-yellow">
              <h3>🟡 Delta WER</h3>
              <div class="val" style="color:{d_color}">{arrow} {abs(delta):.4f}</div>
              <div class="sub">{eval_r['verdict']}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="sf-card sf-card-orange">
              <h3>🟠 % Improvement</h3>
              <div class="val" style="color:{imp_color}">{imp_pct:+.1f}%</div>
              <div class="sub">Grammar correction gain</div>
            </div>""", unsafe_allow_html=True)
    else:
        nc = gram_r["num_corrections"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="sf-card sf-card-blue">
              <h3>🔵 ASR Completed</h3>
              <div class="val">{asr_r['duration_s']}s</div>
              <div class="sub">Whisper · {asr_r['device'].upper()} · {asr_r['language']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            color = "#06D6A0" if nc == 0 else "#FF6B35"
            st.markdown(f"""
            <div class="sf-card sf-card-green">
              <h3>🟢 Corrections Applied</h3>
              <div class="val" style="color:{color}">{nc}</div>
              <div class="sub">LanguageTool fixes</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="sf-callout">
        Tip: Paste the reference / ground-truth text above to unlock WER comparison and the mathematical proof.
        </div>""", unsafe_allow_html=True)

    # Transcripts
    st.markdown('<div class="sf-section">📝 Transcripts</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**🔵 Raw ASR Output** *(Whisper)*")
        st.markdown(f'<div class="tx-box">{asr_r["text"]}</div>', unsafe_allow_html=True)
    with tc2:
        st.markdown("**🟢 Grammar-Corrected** *(LanguageTool)*")
        diff_html = grammar_module.build_diff_html(asr_r["text"], gram_r["corrected_text"])
        st.markdown(f'<div class="tx-box">{diff_html}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sf-callout" style="font-size:0.82rem">
    <span style="background:rgba(239,35,60,0.12);color:#c00;text-decoration:line-through;padding:1px 4px;border-radius:3px">red</span>
    = deleted/replaced &nbsp;·&nbsp;
    <span style="background:rgba(6,214,160,0.18);color:#047a5a;font-weight:700;padding:1px 4px;border-radius:3px">green</span>
    = inserted/corrected
    </div>""", unsafe_allow_html=True)

    # Grammar error detail
    if gram_r["num_corrections"] > 0:
        st.markdown('<div class="sf-section">🔍 Errors Fixed</div>', unsafe_allow_html=True)
        for s in grammar_module.summarise_errors(gram_r["matches"]):
            st.markdown(f"""
            <div class="err-row">
              <span class="rule-chip">{s['rule_id']}</span>
              <span style="color:rgba(0,0,0,0.4);font-size:0.78rem;margin-left:6px">[{s['category']}]</span><br>
              <span>{s['message']}</span><br>
              <span style="color:rgba(0,0,0,0.45);font-size:0.8rem">"{s['context']}"</span>
              &nbsp;&rarr;&nbsp;
              <span style="color:#047a5a;font-weight:700;font-size:0.8rem">"{s['suggestion']}"</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No grammatical errors detected in the ASR transcript for this clip.")

    # WER chart + proof
    if eval_r:
        st.markdown('<div class="sf-section">📈 WER Comparison Chart</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#F5F0E8")

        cats   = ["ASR (Whisper)", "Corrected (LT)"]
        vals   = [eval_r["asr_metrics"]["wer"], eval_r["corrected_metrics"]["wer"]]
        colors = ["#4361EE", "#06D6A0"]

        bars = ax.bar(cats, vals, color=colors, width=0.45,
                      edgecolor="#0D0D0D", linewidth=2.0, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.03,
                    f"{v*100:.2f}%",
                    ha="center", va="bottom",
                    color="#0D0D0D", fontsize=13, fontweight="bold",
                    fontfamily="monospace")

        ax.set_ylim(0, max(vals) * 1.4 + 0.01)
        ax.set_ylabel("Word Error Rate", color="#2A2A2A", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#2A2A2A", labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color("#0D0D0D")
        ax.yaxis.grid(True, color="#CCC", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        if eval_r["delta_wer"] > 0.001:
            ax.annotate(
                f"  down {eval_r['improvement_pct']:.1f}%\n  improvement",
                xy=(1, vals[1]), xytext=(1.38, (vals[0]+vals[1])/2),
                arrowprops=dict(arrowstyle="->", color="#0D0D0D", lw=1.8),
                color="#047a5a", fontsize=9, fontweight="bold",
            )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown('<div class="sf-section">🧮 Mathematical Proof</div>', unsafe_allow_html=True)
        st.markdown(eval_r["math_proof"])


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:0.2rem">
      <span style="font-size:1.6rem">🎙️</span>
      <span style="font-size:1.15rem;font-weight:700;vertical-align:middle"> SpeechFix</span>
    </div>
    <div style="font-size:0.75rem;color:rgba(0,0,0,0.45);text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:1rem">
      ASR · Grammar · Evaluation
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">🤖 Whisper Model</div>', unsafe_allow_html=True)
    model_size = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small"],
        index=1,
        label_visibility="collapsed",
        help="tiny=fastest, small=most accurate"
    )
    st.markdown("""
    <div style="font-size:0.77rem;color:rgba(0,0,0,0.5);line-height:1.7;margin-top:0.3rem">
    🔹 <b>tiny</b> — 39M params, ~2s<br>
    🔹 <b>base</b> — 74M params, ~4s<br>
    🔹 <b>small</b> — 244M, ~10s
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">📂 LJ Speech Dataset (optional)</div>', unsafe_allow_html=True)
    dataset_root = st.text_input(
        "Dataset root",
        value=r"d:\project\LJSpeech-1.1",
        label_visibility="collapsed",
        help="Optional -- leave blank if using upload mode only"
    )
    dataset_ok = Path(dataset_root).is_dir() and (Path(dataset_root) / "metadata.csv").exists()
    if dataset_ok:
        st.success("Dataset found")
    else:
        st.markdown(
            '<div style="font-size:0.78rem;color:rgba(0,0,0,0.4);margin-top:0.3rem">'
            'Dataset not found — upload mode available.</div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">🔢 Batch size</div>', unsafe_allow_html=True)
    batch_n = st.slider("Batch clips", 2, 10, 5, label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sf-hero">
  <h1>Speech<span>Fix</span></h1>
  <p>Audio &rarr; Speech Recognition &rarr; Grammar Correction &rarr; Mathematical Proof</p>
  <div class="pipeline-row">
    <span class="p-badge p-badge-a">🔵 A · Whisper ASR</span>
    <span class="p-badge p-badge-b">🟢 B · LanguageTool</span>
    <span class="p-badge p-badge-c">🟡 C · WER Proof</span>
    <span class="p-badge p-badge-d">🟠 D · This Dashboard</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_upload, tab_dataset, tab_batch, tab_learn = st.tabs([
    "Upload Audio",
    "LJ Speech Dataset",
    "Batch Analysis",
    "How It Works",
])


# ── TAB 1: UPLOAD ─────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown('<div class="sf-section">Upload Any Audio File</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sf-callout">
    Drop any <code>.wav</code> file — no dataset needed.
    Paste a <strong>reference transcript</strong> to unlock WER evaluation and the mathematical proof.
    </div>
    """, unsafe_allow_html=True)

    u_col1, u_col2 = st.columns([1, 1])

    with u_col1:
        uploaded_file = st.file_uploader(
            "Choose a .wav audio file",
            type=["wav"],
            help="Drag & drop or click to browse"
        )
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            st.markdown(f"""
            <div class="sf-card sf-card-yellow" style="margin-top:0.5rem">
              <h3>File info</h3>
              <div style="font-family:'Space Mono',monospace;font-size:0.85rem">
                {uploaded_file.name}<br>
                {uploaded_file.size/1024:.1f} KB
              </div>
            </div>""", unsafe_allow_html=True)

    with u_col2:
        ref_text = st.text_area(
            "Reference transcript (optional)",
            height=130,
            placeholder="Paste ground-truth text here to enable WER calculation...",
            help="The exact words spoken in the audio."
        )

    st.markdown('<hr>', unsafe_allow_html=True)
    run_upload = st.button(
        "Run Pipeline on Uploaded File",
        key="btn_upload",
        disabled=(uploaded_file is None)
    )

    if run_upload and uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            asr_r, gram_r, eval_r = run_pipeline(tmp_path, model_size, ref_text)
            st.success("Pipeline finished!")
            st.divider()
            render_results(asr_r, gram_r, eval_r, ref_text)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── TAB 2: LJ SPEECH DATASET ──────────────────────────────────────────────────
with tab_dataset:
    if not dataset_ok:
        st.markdown("""
        <div class="sf-card sf-card-red" style="margin-top:1rem">
          <h3>Dataset not found</h3>
          <div style="font-size:0.92rem">
            Set the LJ Speech root path in the sidebar, or use the
            <strong>Upload Audio</strong> tab to process any .wav file directly.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        all_records = get_dataset(dataset_root, 200)
        available   = [r for r in all_records if r["exists"]]

        d_col1, d_col2 = st.columns([1.4, 1])
        with d_col1:
            st.markdown('<div class="sf-section">Select Clip</div>', unsafe_allow_html=True)
            labels     = [f"{r['id']}  --  {r['ground_truth'][:60]}..." for r in available]
            chosen_idx = st.selectbox("Clip", range(len(labels)),
                                      format_func=lambda i: labels[i],
                                      label_visibility="collapsed")
            record = available[chosen_idx]
            with open(record["wav_path"], "rb") as f:
                st.audio(f.read(), format="audio/wav")

        with d_col2:
            st.markdown('<div class="sf-section">Ground Truth</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="sf-card sf-card-yellow">
              <h3>Reference -- what was actually spoken</h3>
              <div style="font-size:0.9rem;line-height:1.75">{record['ground_truth']}</div>
              <div class="sub" style="margin-top:0.6rem">
                {record['id']} &nbsp;·&nbsp; {len(record['ground_truth'].split())} words
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)
        run_ds = st.button("Run Full Pipeline", key="btn_dataset")

        if run_ds:
            asr_r, gram_r, eval_r = run_pipeline(
                record["wav_path"], model_size, record["ground_truth"]
            )
            st.success("Pipeline complete!")
            st.divider()
            render_results(asr_r, gram_r, eval_r, record["ground_truth"])


# ── TAB 3: BATCH ──────────────────────────────────────────────────────────────
with tab_batch:
    if not dataset_ok:
        st.info("Batch mode requires the LJ Speech dataset. Set the path in the sidebar.")
    else:
        all_records = get_dataset(dataset_root, 200)
        available   = [r for r in all_records if r["exists"]]
        batch_recs  = available[:batch_n]

        st.markdown("""
        <div class="sf-callout">
        Run the pipeline on <strong>multiple clips</strong> and compare WER across samples.
        </div>""", unsafe_allow_html=True)

        bc1, bc2 = st.columns([1, 3])
        with bc1:
            run_batch = st.button("Run Batch", key="btn_batch")
        with bc2:
            st.markdown(
                f"<div style='padding:0.55rem 0;font-size:0.85rem;color:rgba(0,0,0,0.5)'>"
                f"{len(batch_recs)} clips · ~{len(batch_recs)*6}s estimated</div>",
                unsafe_allow_html=True
            )

        if run_batch:
            rows = []
            prog = st.progress(0, text="Starting...")
            for i, rec in enumerate(batch_recs):
                prog.progress(i / len(batch_recs),
                              text=f"Processing {rec['id']} ({i+1}/{len(batch_recs)})...")
                try:
                    ar = asr_module.transcribe(rec["wav_path"], model_size)
                    gr = grammar_module.correct(ar["text"])
                    er = eval_module.compare(rec["ground_truth"], ar["text"], gr["corrected_text"])
                    rows.append({
                        "ID":            rec["id"],
                        "WER_ASR":       er["asr_metrics"]["wer"],
                        "WER_Corrected": er["corrected_metrics"]["wer"],
                        "ΔWER":          er["delta_wer"],
                        "% Gain":        er["improvement_pct"],
                        "# Fixes":       gr["num_corrections"],
                        "ASR s":         ar["duration_s"],
                    })
                except Exception as e:
                    rows.append({"ID": rec["id"], "WER_ASR": None, "Error": str(e)})

            prog.progress(1.0, text="Done!")
            df = pd.DataFrame(rows)

            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, label, val, acc in [
                (sc1, "Avg ASR WER",       f"{df['WER_ASR'].mean()*100:.2f}%",       "sf-card-blue"),
                (sc2, "Avg Corrected WER", f"{df['WER_Corrected'].mean()*100:.2f}%", "sf-card-green"),
                (sc3, "Avg ΔWER",          f"{df['ΔWER'].mean():+.4f}",              "sf-card-yellow"),
                (sc4, "Avg % Gain",        f"{df['% Gain'].mean():+.1f}%",           "sf-card-orange"),
            ]:
                col.markdown(f"""
                <div class="sf-card {acc}">
                  <h3>{label}</h3>
                  <div class="val">{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sf-section">WER per Clip</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            fig2.patch.set_facecolor("white")
            ax2.set_facecolor("#F5F0E8")
            x = np.arange(len(df))
            w = 0.35
            ax2.bar(x - w/2, df["WER_ASR"],       w, label="ASR",       color="#4361EE", edgecolor="#0D0D0D", linewidth=1.5)
            ax2.bar(x + w/2, df["WER_Corrected"], w, label="Corrected", color="#06D6A0", edgecolor="#0D0D0D", linewidth=1.5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(df["ID"], rotation=30, ha="right")
            ax2.set_ylabel("WER")
            ax2.yaxis.grid(True, color="#ddd", linewidth=0.8)
            for sp in ax2.spines.values():
                sp.set_linewidth(2)
                sp.set_color("#0D0D0D")
            ax2.legend(edgecolor="#0D0D0D")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            st.markdown('<div class="sf-section">Detail Table</div>', unsafe_allow_html=True)
            st.dataframe(
                df[["ID","WER_ASR","WER_Corrected","ΔWER","% Gain","# Fixes","ASR s"]]
                  .style
                  .background_gradient(subset=["WER_ASR","WER_Corrected"], cmap="RdYlGn_r")
                  .background_gradient(subset=["ΔWER","% Gain"],           cmap="RdYlGn")
                  .format({"WER_ASR":"{:.4f}","WER_Corrected":"{:.4f}","ΔWER":"{:+.4f}","% Gain":"{:+.1f}%"}),
                use_container_width=True, hide_index=True,
            )


# ── TAB 4: HOW IT WORKS ───────────────────────────────────────────────────────
with tab_learn:
    st.markdown("## How Each Block Works")

    with st.expander("🔵 Block A — The Ear: Whisper ASR", expanded=True):
        st.markdown("""
**What is ASR?** Converts raw audio waveforms into text.

**Why Whisper?** Trained on **680,000 hours** of audio. Runs 100% locally — no API key.

**Internal flow:**
```
.wav file
  -> Resample to 16,000 Hz mono  (stdlib wave + numpy)
  -> 80-channel log-Mel spectrogram
  -> CNN feature extractor
  -> Transformer Encoder  ->  Transformer Decoder
  -> Byte-Pair Encoding tokens  ->  text output
```

**Common ASR errors:** homophones ("their/there"), mishearing consonants, proper nouns not in vocabulary.
        """)

    with st.expander("🟢 Block B — The Brain: LanguageTool"):
        st.markdown("""
**How LanguageTool works:**
```
Text -> Tokenise -> POS Tag -> Rule Match (4000+ rules) -> Apply Fixes
```

| Rule | Example | Fix |
|------|---------|-----|
| `EN_A_VS_AN` | "a apple" | "an apple" |
| `UPPERCASE_SENTENCE_START` | "hello world" | "Hello world" |
| `MORFOLOGIK_RULE_EN_US` | "recieve" | "receive" |
| `DOUBLE_PUNCTUATION` | "hello.." | "hello." |
        """)

    with st.expander("🟡 Block C — The Judge: WER & Math Proof"):
        st.markdown(r"""
**Word Error Rate:**

$$WER = \frac{S + D + I}{N}$$

$S$ = Substitutions · $D$ = Deletions · $I$ = Insertions · $N$ = Reference word count

Computed via **Levenshtein edit distance** at the word level.

**Proof of improvement:**

$$\Delta WER = WER_{ASR} - WER_{corrected}$$
$$\% Improvement = \frac{\Delta WER}{WER_{ASR}} \times 100$$

$\Delta WER > 0$ means grammar correction moved text **closer** to ground truth.
        """)

    with st.expander("Upload Mode"):
        st.markdown("""
**No dataset required.** Upload any `.wav` file and the pipeline:
1. Saves it to a temporary file
2. Runs Whisper transcription
3. Runs LanguageTool grammar correction
4. Optionally calculates WER if you paste a reference transcript
5. Cleans up the temp file automatically

Supported: any **WAV (PCM)** file — 8-bit, 16-bit, or 32-bit float, mono or stereo.
        """)
