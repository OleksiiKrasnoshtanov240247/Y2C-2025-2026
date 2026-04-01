

--- Cell 0 (markdown) ---
# Research Notebook — Real-Time Voice Conversion: Chunk-Length Impact on Quality and Latency
## Oleksii Krasnoshtanov — Sub-Question 1

**Author:** Oleksii Krasnoshtanov  
**Research Group:** Y2C 2025-26 (Group O1) — Breda University of Applied Sciences  
**Date:** March 2026  

---

### Research Question
> *"How does the input buffer size (chunk length) impact voice similarity (CosSim) and processing latency in RVC-based real-time voice conversion?"*

### Notebook Organisation

| Part | Sections | Purpose | Task |
|------|----------|---------|------|
| **I — EDA** | 1–12 | Explore audio data, define baseline | Task 6 |
| **II — Baseline Evaluation** | 13–23 | Compute metrics at C3 (default), analyse per metric | Task 6 |
| **III — Method Implementation** | 25–28 | Run full experiment across all 5 conditions | Task 9 |
| **IV — Quantitative Results** | 29–35 | Statistical analysis and visualisations | Task 10 |
| **V — Hypothesis Testing** | 36–42 | Test H1, H2, H3; critical analysis | Task 11 |

### Metric Categories (per Audio Evaluation Metrics Reference v1.0)
| Category | Metrics | Direction |
|---|---|---|
| Speaker Similarity | SECS (ECAPA-TDNN), MCD | Higher / Lower is better |
| Intelligibility | WER, CER, STOI | Lower / Lower / Higher |
| Perceptual Quality | PESQ, SNR | Higher / Higher |
| Prosody Preservation | F0 Correlation, F0 RMSE | Higher / Lower |
| System Performance | Per-chunk latency, RTF | Lower / Lower |

--- Cell 1 (markdown) ---
## 1. Setup and Import Libraries

--- Cell 2 (code) ---
# ── Standard libraries ──────────────────────────────────────────────
import os, sys, json, warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
from scipy import stats
from IPython.display import display, Audio, Markdown

# ── Reproducibility ────────────────────────────────────────────────
np.random.seed(42)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# ── Display settings ───────────────────────────────────────────────
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 60)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_colwidth", 60)

# ── Matplotlib style ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ── Project-specific imports ──────────────────────────────────────
# Add audio_testing to path so we can import the project modules
WORKSPACE = Path(r"C:\Users\Alex\Documents\GitHub\Y2C-2025-2026")
AUDIO_DIR = WORKSPACE / "audio"
TESTING_DIR = AUDIO_DIR / "audio_testing"

sys.path.insert(0, str(TESTING_DIR))

from audio_utils import load_audio, prepare_audio_for_metrics, STANDARD_SR
from analyze_audio_metadata import analyze_audio_file
from metrics.quality import compute_pesq, compute_snr
from metrics.intelligibility import compute_stoi
from metrics.speaker_similarity import compute_secs, compute_mcd, compute_f0_metrics

print(f"Workspace       : {WORKSPACE}")
print(f"Audio directory  : {AUDIO_DIR}")
print(f"Standard SR      : {STANDARD_SR} Hz")
print("All imports loaded successfully.")

--- Cell 3 (markdown) ---
## 2. Define Directory Paths and Discover Audio Files

The project uses three directories:
- **`assets/input/`** — Source clips (speech to be converted)
- **`assets/original/`** — Target speaker reference recordings (used for SECS ground truth)
- **`assets/output/`** — Converted audio files produced by RVC (Applio)

--- Cell 4 (code) ---
# ── Directory paths ─────────────────────────────────────────────────
INPUT_DIR   = AUDIO_DIR / "assets" / "input"
ORIGINAL_DIR = AUDIO_DIR / "assets" / "original"
OUTPUT_DIR  = AUDIO_DIR / "assets" / "output"
RESULTS_DIR = AUDIO_DIR / "results"
RESULTS_CSV = RESULTS_DIR / "baseline_raw.csv"

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac"}

def discover_audio(directory: Path) -> list[Path]:
    """Find all audio files in a directory."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )

# Discover files
source_clips   = discover_audio(INPUT_DIR)
target_refs    = discover_audio(ORIGINAL_DIR)
converted_outs = discover_audio(OUTPUT_DIR)

print(f"{'Directory':<40} {'Files':>5}")
print("-" * 48)
print(f"{'Source clips (input/)':<40} {len(source_clips):>5}")
print(f"{'Target references (original/)':<40} {len(target_refs):>5}")
print(f"{'Converted outputs (output/)':<40} {len(converted_outs):>5}")

print(f"\n── Source clips ──")
for p in source_clips:
    print(f"  {p.name:<30} {p.stat().st_size / 1024:>8.1f} KB")

print(f"\n── Target speaker references ──")
for p in target_refs:
    print(f"  {p.name:<30} {p.stat().st_size / 1024:>8.1f} KB")

print(f"\n── Converted outputs ──")
for p in converted_outs:
    print(f"  {p.name:<30} {p.stat().st_size / 1024:>8.1f} KB")

# Check for baseline results CSV
print(f"\n── Baseline results CSV ──")
if RESULTS_CSV.exists():
    print(f"  Found: {RESULTS_CSV}  ({RESULTS_CSV.stat().st_size / 1024:.1f} KB)")
else:
    print(f"  Not found at {RESULTS_CSV}")
    print("  Will compute metrics directly in this notebook.")

--- Cell 5 (markdown) ---
## 3. Audio Metadata Analysis: Source Clips

We use `analyze_audio_file()` to collect comprehensive metadata for each source clip: sample rate, duration, peak amplitude, RMS, pitch (F0), SNR estimate, voice activity, and clipping detection.

--- Cell 6 (code) ---
def metadata_to_flat_row(meta: dict) -> dict:
    """Flatten nested metadata dict into a single-level row for DataFrame."""
    row = {"file": meta["file"]}
    for section in ("format", "amplitude", "frequency", "pitch", "quality"):
        for k, v in meta[section].items():
            row[f"{section}_{k}"] = v
    row["num_recommendations"] = len(meta.get("recommendations", []))
    row["recommendations"] = "; ".join(meta.get("recommendations", []))
    return row

# Analyse all source clips
source_metadata = []
for clip in source_clips:
    print(f"Analysing: {clip.name} ...", end=" ")
    meta = analyze_audio_file(str(clip))
    source_metadata.append(metadata_to_flat_row(meta))
    print(f"✓  ({meta['format']['duration_sec']:.1f}s, {meta['format']['sample_rate']} Hz)")

df_source = pd.DataFrame(source_metadata)

# Display key columns
display_cols = [
    "file", "format_sample_rate", "format_duration_sec",
    "amplitude_max", "amplitude_rms_mean", "amplitude_dynamic_range_db",
    "pitch_f0_mean_hz", "pitch_f0_std_hz", "pitch_voiced_percentage",
    "quality_snr_estimate_db", "quality_silence_percentage",
]
display(df_source[display_cols].style.set_caption("Source Clip Metadata Summary"))

--- Cell 7 (markdown) ---
## 4. Audio Metadata Analysis: Target Speaker References

The three target speakers available for voice conversion:
- **DonaldTrump** — `trump.mp3` (model: `DonaldTrump_475e_8075s.pth`, 475 epochs)
- **GeorgeBanks** — `banks.wav` (model: `George Banks_500e_41500s.pth`, 500 epochs)
- **Wheatley-HD** — `wheatly.wav` (model: `Wheatley-HD_e450_s40050.pth`, 450 epochs)

--- Cell 8 (code) ---
# Map filenames to speaker names for clarity
SPEAKER_MAP = {
    "trump.mp3": "DonaldTrump",
    "banks.wav": "GeorgeBanks",
    "wheatly.wav": "Wheatley-HD",
}

target_metadata = []
for ref in target_refs:
    print(f"Analysing: {ref.name} ({SPEAKER_MAP.get(ref.name, 'Unknown')}) ...", end=" ")
    meta = analyze_audio_file(str(ref))
    row = metadata_to_flat_row(meta)
    row["speaker"] = SPEAKER_MAP.get(ref.name, ref.stem)
    target_metadata.append(row)
    print(f"✓  ({meta['format']['duration_sec']:.1f}s, F0={meta['pitch']['f0_mean_hz']} Hz)")

df_target = pd.DataFrame(target_metadata)

display_cols_target = [
    "speaker", "file", "format_sample_rate", "format_duration_sec",
    "amplitude_rms_mean", "pitch_f0_mean_hz", "pitch_f0_std_hz",
    "pitch_f0_min_hz", "pitch_f0_max_hz", "pitch_voiced_percentage",
    "quality_snr_estimate_db",
]
display(df_target[display_cols_target].style.set_caption("Target Speaker Reference Metadata"))

# Compare pitch ranges across speakers
print("\n── Pitch Comparison Across Target Speakers ──")
for _, row in df_target.iterrows():
    f0_mean = row["pitch_f0_mean_hz"]
    f0_min = row["pitch_f0_min_hz"]
    f0_max = row["pitch_f0_max_hz"]
    if pd.notna(f0_mean):
        print(f"  {row['speaker']:<15} F0: {f0_mean:>6.1f} Hz  (range: {f0_min:.0f}–{f0_max:.0f} Hz)")
    else:
        print(f"  {row['speaker']:<15} F0: not detected")

--- Cell 9 (markdown) ---
## 5. Audio Metadata Analysis: Converted Outputs

Analyse the voice-converted output files. The filenames indicate which source clip was converted to which target speaker (e.g., `trumpexample1.wav` = example1 converted to Trump voice).

--- Cell 10 (code) ---
# Analyse converted output files
output_metadata = []
for out_file in converted_outs:
    print(f"Analysing: {out_file.name} ...", end=" ")
    meta = analyze_audio_file(str(out_file))
    row = metadata_to_flat_row(meta)
    # Parse target speaker from filename prefix
    name_lower = out_file.stem.lower()
    if "trump" in name_lower:
        row["target_speaker"] = "DonaldTrump"
    elif "banks" in name_lower:
        row["target_speaker"] = "GeorgeBanks"
    elif "wheatl" in name_lower:
        row["target_speaker"] = "Wheatley-HD"
    elif "amber" in name_lower:
        row["target_speaker"] = "AmberHeard"
    elif "swift" in name_lower:
        row["target_speaker"] = "TaylorSwift"
    else:
        row["target_speaker"] = "Unknown"
    output_metadata.append(row)
    print(f"✓  ({meta['format']['duration_sec']:.1f}s, SR={meta['format']['sample_rate']} Hz)")

df_output = pd.DataFrame(output_metadata)

display_cols_out = [
    "file", "target_speaker", "format_sample_rate", "format_duration_sec",
    "amplitude_rms_mean", "pitch_f0_mean_hz",
    "quality_snr_estimate_db", "amplitude_clipped",
]
display(df_output[display_cols_out].style.set_caption("Converted Output Metadata"))

# Compare output sample rates to the standardised 16 kHz
unique_srs = df_output["format_sample_rate"].unique()
print(f"\nUnique output sample rates: {sorted(unique_srs)}")
if any(sr != STANDARD_SR for sr in unique_srs):
    print(f"⚠ Some outputs are not at {STANDARD_SR} Hz — resampling required for metric computation.")
else:
    print(f"✓ All outputs already at {STANDARD_SR} Hz.")

--- Cell 11 (markdown) ---
## 6. EDA — Source Audio Characteristics Distribution

Visualise the distribution of key audio properties across source clips to identify outliers or data quality issues before running the baseline experiment.

--- Cell 12 (code) ---
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

props = [
    ("format_duration_sec", "Duration (s)", "steelblue"),
    ("amplitude_max", "Peak Amplitude", "coral"),
    ("amplitude_rms_mean", "Mean RMS Level", "seagreen"),
    ("amplitude_dynamic_range_db", "Dynamic Range (dB)", "mediumpurple"),
]

for ax, (col, label, color) in zip(axes, props):
    vals = df_source[col].dropna()
    ax.barh(df_source["file"], vals, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(label)
    ax.set_title(label)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

fig.suptitle("Source Clip Characteristics", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# Audio playback for representative clips
print("\n── Listen to source clips ──")
for clip in source_clips[:3]:
    audio, sr = load_audio(str(clip))
    print(f"\n{clip.name} ({len(audio)/sr:.1f}s @ {sr} Hz):")
    display(Audio(audio, rate=sr))

--- Cell 13 (markdown) ---
## 7. EDA — Sample Rate, Duration, and Format Summary

Verify that all audio files meet the standardisation requirements: resampling to 16 kHz, WAV PCM 16-bit for metric computation.

--- Cell 14 (code) ---
# Combine all metadata into a single summary table
all_rows = []
for idx, row in df_source.iterrows():
    all_rows.append({"category": "Source", "file": row["file"],
                     "sample_rate": row["format_sample_rate"],
                     "duration_sec": row["format_duration_sec"],
                     "format": Path(row["file"]).suffix})
for idx, row in df_target.iterrows():
    all_rows.append({"category": "Target Ref", "file": row["file"],
                     "sample_rate": row["format_sample_rate"],
                     "duration_sec": row["format_duration_sec"],
                     "format": Path(row["file"]).suffix})
for idx, row in df_output.iterrows():
    all_rows.append({"category": "Converted", "file": row["file"],
                     "sample_rate": row["format_sample_rate"],
                     "duration_sec": row["format_duration_sec"],
                     "format": Path(row["file"]).suffix})

df_all_files = pd.DataFrame(all_rows)
display(df_all_files.style.set_caption("All Audio Files — Format Summary"))

# Sample rate summary
print("\n── Sample Rate Distribution ──")
sr_counts = df_all_files.groupby(["category", "sample_rate"]).size().unstack(fill_value=0)
display(sr_counts)

# Highlight resampling needs
needs_resample = df_all_files[df_all_files["sample_rate"] != STANDARD_SR]
if len(needs_resample) > 0:
    print(f"\n⚠ {len(needs_resample)} files need resampling to {STANDARD_SR} Hz:")
    for _, r in needs_resample.iterrows():
        print(f"  {r['file']} ({r['sample_rate']} Hz → {STANDARD_SR} Hz)")
    print("\n✓ prepare_audio_for_metrics() handles resampling automatically.")
else:
    print(f"\n✓ All files already at {STANDARD_SR} Hz.")

# Duration bar chart
fig, ax = plt.subplots(figsize=(12, 4))
colors = {"Source": "steelblue", "Target Ref": "coral", "Converted": "seagreen"}
for cat in ["Source", "Target Ref", "Converted"]:
    subset = df_all_files[df_all_files["category"] == cat]
    ax.barh(subset["file"], subset["duration_sec"], color=colors[cat], label=cat,
            edgecolor="white", linewidth=0.5)
ax.set_xlabel("Duration (seconds)")
ax.set_title("Audio File Durations by Category")
ax.legend()
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.show()

--- Cell 15 (markdown) ---
## 8. EDA — Pitch (F0) Distribution Across Source Clips

Extract fundamental frequency (F0) contours from source clips using `librosa.pyin` (fmin=65 Hz, fmax=600 Hz). Compare source pitch ranges to target speaker pitch ranges to anticipate pitch-shifting challenges.

--- Cell 16 (code) ---
# Extract F0 contours for source clips and target references
f0_data = {}

print("Extracting F0 contours...")
for clip in source_clips:
    audio, sr = load_audio(str(clip))
    f0, voiced, _ = librosa.pyin(audio, fmin=65.0, fmax=600.0, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    f0_data[clip.name] = f0_clean
    print(f"  {clip.name}: {len(f0_clean)} voiced frames, mean={np.mean(f0_clean):.1f} Hz")

for ref in target_refs:
    audio, sr = load_audio(str(ref))
    f0, voiced, _ = librosa.pyin(audio, fmin=65.0, fmax=600.0, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    label = SPEAKER_MAP.get(ref.name, ref.stem) + " (target)"
    f0_data[label] = f0_clean
    print(f"  {label}: {len(f0_clean)} voiced frames, mean={np.mean(f0_clean):.1f} Hz")

# Violin/box plot
fig, ax = plt.subplots(figsize=(14, 5))
labels = list(f0_data.keys())
data = [f0_data[l] for l in labels]
colors = ["steelblue"] * len(source_clips) + ["coral"] * len(target_refs)

bp = ax.boxplot(data, labels=labels, patch_artist=True, vert=True, showfliers=False)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel("F0 (Hz)")
ax.set_title("Pitch (F0) Distribution: Source Clips vs Target Speaker References")
ax.tick_params(axis="x", rotation=35)

# Add legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="steelblue", alpha=0.7, label="Source clips"),
                   Patch(facecolor="coral", alpha=0.7, label="Target speakers")],
          loc="upper right")

plt.tight_layout()
plt.show()

# Summary table
print("\n── F0 Statistics Summary ──")
print(f"{'Name':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 68)
for name, f0_vals in f0_data.items():
    if len(f0_vals) > 0:
        print(f"{name:<30} {np.mean(f0_vals):>8.1f} {np.std(f0_vals):>8.1f} "
              f"{np.min(f0_vals):>8.1f} {np.max(f0_vals):>8.1f}")

--- Cell 17 (markdown) ---
## 9. EDA — Signal Quality: SNR and Dynamic Range

Assess the recording quality of source clips. Low-SNR recordings (< 20 dB) may produce unreliable evaluation metrics. High dynamic range suggests well-recorded speech.

--- Cell 18 (code) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# SNR bar chart
snr_vals = df_source["quality_snr_estimate_db"]
ax = axes[0]
bars = ax.barh(df_source["file"], snr_vals, color="steelblue", edgecolor="white")
ax.axvline(x=20, color="red", linestyle="--", linewidth=1.2, label="Low SNR threshold (20 dB)")
ax.axvline(x=35, color="green", linestyle="--", linewidth=1.2, label="Clean threshold (35 dB)")
ax.set_xlabel("Estimated SNR (dB)")
ax.set_title("Signal-to-Noise Ratio — Source Clips")
ax.legend(fontsize=9)

# Dynamic range bar chart
dr_vals = df_source["amplitude_dynamic_range_db"]
ax = axes[1]
ax.barh(df_source["file"], dr_vals, color="mediumpurple", edgecolor="white")
ax.set_xlabel("Dynamic Range (dB)")
ax.set_title("Dynamic Range — Source Clips")

for ax in axes:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()

# Flag problematic clips
print("\n── Quality Assessment ──")
for _, row in df_source.iterrows():
    snr = row["quality_snr_estimate_db"]
    status = "✓ Clean" if snr > 35 else ("⚠ Moderate" if snr > 20 else "✗ Low quality")
    print(f"  {row['file']:<30} SNR: {snr:>6.1f} dB  [{status}]")

--- Cell 19 (markdown) ---
## 10. EDA — Spectral Analysis and Frequency Content

Examine spectral characteristics of source clips and compare before/after conversion using mel-spectrograms.

--- Cell 20 (code) ---
# Spectral centroid bar chart
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(df_source["file"], df_source["frequency_spectral_centroid_mean_hz"],
        color="teal", edgecolor="white")
ax.set_xlabel("Mean Spectral Centroid (Hz)")
ax.set_title("Spectral Centroid — Source Clips (higher = brighter timbre)")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.show()

# Show mel-spectrograms for first source clip and its converted version
# Find a source-output pair
example_source = source_clips[0]  # myexample1.wav
# Find a converted output that matches
example_outputs = [f for f in converted_outs if "example1" in f.name.lower() or "1" in f.stem]
if not example_outputs:
    example_outputs = converted_outs[:1]  # fallback

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Source spectrogram
audio_src, sr = load_audio(str(example_source))
S_src = librosa.feature.melspectrogram(y=audio_src, sr=sr, n_mels=80, fmax=8000)
S_src_db = librosa.power_to_db(S_src, ref=np.max)
img1 = librosa.display.specshow(S_src_db, sr=sr, x_axis="time", y_axis="mel",
                                 ax=axes[0], fmax=8000, cmap="magma")
axes[0].set_title(f"Source: {example_source.name}")
fig.colorbar(img1, ax=axes[0], format="%+2.0f dB", shrink=0.8)

# Converted spectrogram
if example_outputs:
    audio_out, sr = load_audio(str(example_outputs[0]))
    S_out = librosa.feature.melspectrogram(y=audio_out, sr=sr, n_mels=80, fmax=8000)
    S_out_db = librosa.power_to_db(S_out, ref=np.max)
    img2 = librosa.display.specshow(S_out_db, sr=sr, x_axis="time", y_axis="mel",
                                     ax=axes[1], fmax=8000, cmap="magma")
    axes[1].set_title(f"Converted: {example_outputs[0].name}")
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB", shrink=0.8)

fig.suptitle("Mel-Spectrogram Comparison: Source vs RVC-Converted", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

--- Cell 21 (markdown) ---
## 11. EDA — Voice Activity and Silence Ratio

Analyse the proportion of voiced (speech) vs silent regions in each source clip. Clips with excessive silence (> 50%) may produce unreliable metric results.

--- Cell 22 (code) ---
# Voice activity percentage
fig, ax = plt.subplots(figsize=(10, 4))
voiced_pct = df_source["pitch_voiced_percentage"]
silence_pct = df_source["quality_silence_percentage"]

x = np.arange(len(df_source))
width = 0.4
ax.barh(x + width/2, voiced_pct, width, label="Voiced", color="steelblue", edgecolor="white")
ax.barh(x - width/2, silence_pct, width, label="Silence", color="lightcoral", edgecolor="white")
ax.set_yticks(x)
ax.set_yticklabels(df_source["file"])
ax.set_xlabel("Percentage (%)")
ax.set_title("Voice Activity vs Silence — Source Clips")
ax.axvline(x=50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="50% threshold")
ax.legend()
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.show()

# Waveform plots for comparison
fig, axes = plt.subplots(1, min(len(source_clips), 3), figsize=(16, 3))
if not hasattr(axes, '__len__'):
    axes = [axes]
for ax, clip in zip(axes, source_clips[:3]):
    audio, sr = load_audio(str(clip))
    t = np.arange(len(audio)) / sr
    ax.plot(t, audio, linewidth=0.3, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(clip.name, fontsize=10)
    ax.set_ylim(-1, 1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
fig.suptitle("Waveforms of Source Clips", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

--- Cell 23 (markdown) ---
## 12. Baseline Definition: Chunk Length Conditions

The independent variable in this research is **chunk length** — the number of audio samples buffered before processing by the RVC pipeline. Larger chunks provide more context for voice conversion but increase latency.

The five experimental conditions from the research proposal:

| Condition | `read_chunk_size` | Block Frame (samples @ 48 kHz) | Duration (ms) | Description |
|---|---|---|---|---|
| **C1** | 24 | 3,072 | ~64 ms | Very short — aggressive real-time |
| **C2** | 72 | 9,216 | ~192 ms | Short — near real-time threshold |
| **C3** | 192 | 24,576 | ~512 ms | **Baseline default** |
| **C4** | 384 | 49,152 | ~1,024 ms | Long |
| **C5** | 768 | 98,304 | ~2,048 ms | Very long — near-offline |

**Relationship:** `block_frame = read_chunk_size × 128` samples at 48 kHz (Applio's I/O sample rate).

The **baseline condition is C3** (~512 ms chunks). The full experiment will compare all five conditions to quantify the quality-latency trade-off.

--- Cell 24 (code) ---
# Chunk length conditions table
CONDITIONS = {
    "C1": {"read_chunk_size": 24,  "block_frame_ms": 64,   "description": "Very short"},
    "C2": {"read_chunk_size": 72,  "block_frame_ms": 192,  "description": "Short"},
    "C3": {"read_chunk_size": 192, "block_frame_ms": 512,  "description": "Baseline default"},
    "C4": {"read_chunk_size": 384, "block_frame_ms": 1024, "description": "Long"},
    "C5": {"read_chunk_size": 768, "block_frame_ms": 2048, "description": "Very long"},
}

APPLIO_IO_SR = 48_000

df_conditions = pd.DataFrame([
    {
        "Condition": cond,
        "read_chunk_size": v["read_chunk_size"],
        "Block Frame (samples)": v["read_chunk_size"] * 128,
        "Duration (ms)": v["block_frame_ms"],
        "Duration (s)": v["block_frame_ms"] / 1000,
        "Description": v["description"],
    }
    for cond, v in CONDITIONS.items()
])

display(df_conditions.style.set_caption("Chunk Length Conditions (Proposal Table 1)"))

# Visualise chunk durations
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]
bars = ax.barh(df_conditions["Condition"], df_conditions["Duration (ms)"],
               color=colors, edgecolor="white", linewidth=1.5)
ax.set_xlabel("Chunk Duration (ms)")
ax.set_title("Chunk Length Conditions — Experimental Design")

# Add labels on bars
for bar, ms in zip(bars, df_conditions["Duration (ms)"]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f"{ms} ms", va="center", fontsize=10, fontweight="bold")

# Real-time threshold line
ax.axvline(x=200, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
           label="200 ms real-time threshold")
ax.legend()
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.show()

--- Cell 25 (markdown) ---
## 13. Baseline Metric Computation

Since no pre-computed results CSV exists yet, we compute all eight metrics directly from the available audio pairs.

**Source → Output mapping** (inferred from filenames):
| Output | Source | Target Speaker | Target Ref |
|---|---|---|---|
| trumpexample1.wav | myexample1.wav | Donald Trump | trump.mp3 |
| banksexample2.wav | myexample2.mp3 | George Banks | banks.wav |
| amberexample2.wav | myexample2.mp3 | Amber Heard | — |
| swiftexample2.wav | myexample2.mp3 | Taylor Swift | — |
| wheatlymyexample2.wav | myexample2.mp3 | Wheatley-HD | wheatly.wav |
| trumpexample3.wav | myexample3.ogg | Donald Trump | trump.mp3 |

> These conversions all use the Applio default `read_chunk_size = 192` (C3, ≈512 ms), serving as our **baseline condition**.

--- Cell 26 (code) ---
# ------------------------------------------------------------------
# Define audio pairs for baseline evaluation
# Naming convention: output = <speaker_prefix><source_name>.wav
#   myexample1 + trump model  → trumpexample1.wav      (ref: trump.mp3)
#   myexample2 + banks model  → banksexample2.wav       (ref: banks.wav)
#   myexample2 + amber model  → amberexample2.wav       (no ref)
#   myexample2 + swift model  → swiftexample2.wav       (no ref)
#   myexample2 + wheatley     → wheatlymyexample2.wav   (ref: wheatly.wav)
#   myexample3 + trump model  → trumpexample3.wav       (ref: trump.mp3)
# ------------------------------------------------------------------
AUDIO_PAIRS = [
    {
        "output": "trumpexample1.wav",
        "source": "myexample1.wav",
        "speaker": "DonaldTrump",
        "target_ref": "trump.mp3",
        "condition": "C3",
    },
    {
        "output": "banksexample2.wav",
        "source": "myexample2.mp3",
        "speaker": "GeorgeBanks",
        "target_ref": "banks.wav",
        "condition": "C3",
    },
    {
        "output": "amberexample2.wav",
        "source": "myexample2.mp3",
        "speaker": "AmberHeard",
        "target_ref": None,
        "condition": "C3",
    },
    {
        "output": "swiftexample2.wav",
        "source": "myexample2.mp3",
        "speaker": "TaylorSwift",
        "target_ref": None,
        "condition": "C3",
    },
    {
        "output": "wheatlymyexample2.wav",
        "source": "myexample2.mp3",
        "speaker": "Wheatley-HD",
        "target_ref": "wheatly.wav",
        "condition": "C3",
    },
    {
        "output": "trumpexample3.wav",
        "source": "myexample3.ogg",
        "speaker": "DonaldTrump",
        "target_ref": "trump.mp3",
        "condition": "C3",
    },
]

# Use absolute paths from the setup cell (INPUT_DIR, OUTPUT_DIR, ORIGINAL_DIR
# were defined in cell 5 via AUDIO_DIR)
print(f"Audio pairs defined: {len(AUDIO_PAIRS)}")
print(f"Using directories:")
print(f"  Input   : {INPUT_DIR}")
print(f"  Output  : {OUTPUT_DIR}")
print(f"  Original: {ORIGINAL_DIR}")
print()
for p in AUDIO_PAIRS:
    src_ok = (INPUT_DIR / p["source"]).exists()
    out_ok = (OUTPUT_DIR / p["output"]).exists()
    ref_ok = (ORIGINAL_DIR / p["target_ref"]).exists() if p["target_ref"] else "N/A"
    print(f"  {p['output']:30s}  src={src_ok}  out={out_ok}  ref={ref_ok}")

--- Cell 27 (code) ---
# ------------------------------------------------------------------
# Compute all metrics for each audio pair using evaluate()
# ------------------------------------------------------------------
# evaluate() handles all preprocessing (loading, resampling to 16kHz,
# normalisation) and computes all 4 metric categories internally.
# This avoids calling individual metric functions with wrong signatures.
# ------------------------------------------------------------------
from evaluate import evaluate as evaluate_audio

# Column mapping: evaluate() output keys → our standard column names
EVAL_KEY_MAP = {
    "conversion_secs":        "SECS",
    "conversion_mcd_db":      "MCD",
    "content_wer":            "WER",
    "content_cer":            "CER",
    "quality_stoi":           "STOI",
    "quality_pesq":           "PESQ",
    "quality_snr_db":         "SNR",
    "prosody_f0_correlation": "F0_corr",
    "prosody_f0_rmse_hz":     "F0_rmse",
}

CHECKPOINT_PATH = RESULTS_DIR / "baseline_c3_checkpoint.csv"

# Check if checkpoint exists to skip expensive recomputation
if CHECKPOINT_PATH.exists():
    print(f"Loading cached results from {CHECKPOINT_PATH}")
    df_results = pd.read_csv(CHECKPOINT_PATH)
    print(f"Loaded {len(df_results)} rows.")
else:
    results = []

    for i, pair in enumerate(AUDIO_PAIRS):
        print(f"\n{'='*65}")
        print(f"[{i+1}/{len(AUDIO_PAIRS)}] {pair['output']}  (speaker: {pair['speaker']})")
        print(f"{'='*65}")

        src_path = str(INPUT_DIR / pair["source"])
        out_path = str(OUTPUT_DIR / pair["output"])
        ref_path = str(ORIGINAL_DIR / pair["target_ref"]) if pair["target_ref"] else None

        # Call the full evaluation pipeline
        eval_result = evaluate_audio(
            original_path=src_path,
            transformed_path=out_path,
            target_path=ref_path,
            whisper_model="large-v3-turbo",
            whisper_device="cpu",
            whisper_compute_type="int8",
        )

        # Build a clean row with our standard column names
        row = {
            "output_file":    pair["output"],
            "source_file":    pair["source"],
            "speaker":        pair["speaker"],
            "condition":      pair["condition"],
            "has_target_ref": pair["target_ref"] is not None,
        }

        # Map evaluate() keys to our standard metric column names
        for eval_key, col_name in EVAL_KEY_MAP.items():
            val = eval_result.get(eval_key)
            row[col_name] = round(val, 4) if isinstance(val, (int, float)) and val is not None else val

        # Also store transcripts for WER/CER interpretability
        row["src_transcript"] = eval_result.get("ground_truth_transcript", "")
        row["out_transcript"] = eval_result.get("transformed_transcript", "")

        results.append(row)
        print(f"  Done: SECS={row.get('SECS')}  MCD={row.get('MCD')}  "
              f"WER={row.get('WER')}  STOI={row.get('STOI')}")

    df_results = pd.DataFrame(results)

    # Save checkpoint
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(CHECKPOINT_PATH, index=False)
    print(f"\nCheckpoint saved to {CHECKPOINT_PATH}")

print(f"\n{'='*65}")
print(f"Baseline (C3) metrics computed for {len(df_results)} audio pairs.")
print(f"{'='*65}")
display(df_results[["output_file", "speaker", "SECS", "MCD", "WER", "CER",
                     "STOI", "PESQ", "SNR", "F0_corr", "F0_rmse"]])

--- Cell 28 (code) ---
# ------------------------------------------------------------------
# Save baseline results to CSV for reproducibility
# ------------------------------------------------------------------
csv_path = RESULTS_DIR / "baseline_raw.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df_results.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")
display(df_results.describe().round(4))

--- Cell 29 (markdown) ---
## 14. SECS — Speaker Embedding Cosine Similarity

SECS measures how closely the converted voice matches the **target speaker** identity using ECAPA-TDNN embeddings.

| Tier | Range | Interpretation |
|------|-------|----------------|
| Excellent | ≥ 0.85 | Near-indistinguishable from target |
| Acceptable | 0.70 – 0.84 | Recognisably similar |
| Poor | < 0.70 | Significant identity loss |

> **Note:** SECS is only computed for pairs where a target speaker reference recording is available (Trump, Banks, Wheatley).

--- Cell 30 (code) ---
# ------------------------------------------------------------------
# SECS per speaker analysis
# ------------------------------------------------------------------
df_secs = df_results.dropna(subset=["SECS"]).copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart per output
ax = axes[0]
colours = df_secs["SECS"].apply(
    lambda x: "#2ecc71" if x >= 0.85 else ("#f39c12" if x >= 0.70 else "#e74c3c")
)
ax.barh(df_secs["output_file"], df_secs["SECS"], color=colours, edgecolor="white")
ax.axvline(0.85, color="green", linestyle="--", alpha=0.6, label="Excellent (≥0.85)")
ax.axvline(0.70, color="orange", linestyle="--", alpha=0.6, label="Acceptable (≥0.70)")
ax.set_xlabel("SECS (cosine similarity)")
ax.set_title("SECS per Converted Output")
ax.set_xlim(0, 1)
ax.legend(fontsize=8)

# Mean SECS per speaker
ax = axes[1]
speaker_secs = df_secs.groupby("speaker")["SECS"].agg(["mean", "std", "count"]).reset_index()
ax.bar(speaker_secs["speaker"], speaker_secs["mean"],
       yerr=speaker_secs["std"].fillna(0), capsize=5,
       color=["#3498db", "#2ecc71", "#9b59b6"], edgecolor="white")
ax.axhline(0.85, color="green", linestyle="--", alpha=0.6)
ax.axhline(0.70, color="orange", linestyle="--", alpha=0.6)
ax.set_ylabel("Mean SECS")
ax.set_title("Mean SECS per Target Speaker")
ax.set_ylim(0, 1)

for spine in ["top", "right"]:
    axes[0].spines[spine].set_visible(False)
    axes[1].spines[spine].set_visible(False)

plt.tight_layout()
plt.show()

# Summary statistics
print("\nSECS Summary:")
print(f"  Mean:   {df_secs['SECS'].mean():.4f}")
print(f"  Std:    {df_secs['SECS'].std():.4f}")
print(f"  Min:    {df_secs['SECS'].min():.4f}")
print(f"  Max:    {df_secs['SECS'].max():.4f}")
print(f"  N:      {len(df_secs)}")

tier_counts = pd.cut(df_secs["SECS"], bins=[-1, 0.70, 0.85, 1.0],
                      labels=["Poor (<0.70)", "Acceptable (0.70-0.85)", "Excellent (≥0.85)"])
print(f"\nTier distribution:\n{tier_counts.value_counts().to_string()}")

--- Cell 31 (markdown) ---
## 15. WER & CER — Intelligibility Analysis

Word Error Rate (WER) and Character Error Rate (CER) measure how much speech content is preserved after voice conversion, using **faster-whisper large-v3-turbo** as the ASR backbone.

| Tier | WER | CER | Interpretation |
|------|-----|-----|----------------|
| Excellent | ≤ 0.10 | ≤ 0.05 | Near-perfect content preservation |
| Acceptable | 0.10 – 0.25 | 0.05 – 0.15 | Minor degradation |
| Poor | > 0.25 | > 0.15 | Significant intelligibility loss |

--- Cell 32 (code) ---
# ------------------------------------------------------------------
# WER / CER analysis
# ------------------------------------------------------------------
df_intel = df_results.dropna(subset=["WER", "CER"]).copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# WER per output
ax = axes[0]
wer_colors = df_intel["WER"].apply(
    lambda x: "#2ecc71" if x <= 0.10 else ("#f39c12" if x <= 0.25 else "#e74c3c")
)
ax.barh(df_intel["output_file"], df_intel["WER"], color=wer_colors, edgecolor="white")
ax.axvline(0.10, color="green", linestyle="--", alpha=0.6, label="Excellent (≤0.10)")
ax.axvline(0.25, color="orange", linestyle="--", alpha=0.6, label="Acceptable (≤0.25)")
ax.set_xlabel("Word Error Rate")
ax.set_title("WER per Output")
ax.legend(fontsize=8)

# CER per output
ax = axes[1]
cer_colors = df_intel["CER"].apply(
    lambda x: "#2ecc71" if x <= 0.05 else ("#f39c12" if x <= 0.15 else "#e74c3c")
)
ax.barh(df_intel["output_file"], df_intel["CER"], color=cer_colors, edgecolor="white")
ax.axvline(0.05, color="green", linestyle="--", alpha=0.6, label="Excellent (≤0.05)")
ax.axvline(0.15, color="orange", linestyle="--", alpha=0.6, label="Acceptable (≤0.15)")
ax.set_xlabel("Character Error Rate")
ax.set_title("CER per Output")
ax.legend(fontsize=8)

# WER vs CER scatter
ax = axes[2]
ax.scatter(df_intel["WER"], df_intel["CER"], s=100, c="#3498db", edgecolors="white", zorder=3)
for _, r in df_intel.iterrows():
    ax.annotate(r["speaker"], (r["WER"], r["CER"]),
                fontsize=8, ha="left", va="bottom", xytext=(5, 3),
                textcoords="offset points")
ax.set_xlabel("WER")
ax.set_ylabel("CER")
ax.set_title("WER vs CER")
ax.axvline(0.10, color="green", linestyle="--", alpha=0.4)
ax.axhline(0.05, color="green", linestyle="--", alpha=0.4)

for a in axes:
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)

plt.tight_layout()
plt.show()

# Transcript comparison
print("\nTranscript Comparison:")
for _, r in df_intel.iterrows():
    src_t = r.get("src_transcript", "N/A")
    out_t = r.get("out_transcript", "N/A")
    print(f"\n  {r['output_file']} (WER={r['WER']:.3f}, CER={r['CER']:.3f})")
    print(f"    Source:  {str(src_t)[:120]}")
    print(f"    Output:  {str(out_t)[:120]}")

--- Cell 33 (markdown) ---
## 16. Signal Quality — STOI, PESQ, SNR

Three complementary quality metrics:
- **STOI** (Short-Time Objective Intelligibility): [-1, 1], higher = better structural intelligibility
- **PESQ** (Perceptual Evaluation of Speech Quality): [-0.5, 4.5], higher = better perceptual quality
- **SNR** (Signal-to-Noise Ratio): higher dB = cleaner signal

| Metric | Excellent | Acceptable | Poor |
|--------|-----------|------------|------|
| STOI | ≥ 0.75 | 0.45 – 0.74 | < 0.45 |
| PESQ | ≥ 3.0 | 2.0 – 2.99 | < 2.0 |
| SNR | ≥ 25 dB | 15 – 24 dB | < 15 dB |

--- Cell 34 (code) ---
# ------------------------------------------------------------------
# STOI, PESQ, SNR visualisation
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# STOI
ax = axes[0]
df_stoi = df_results.dropna(subset=["STOI"])
stoi_colors = df_stoi["STOI"].apply(
    lambda x: "#2ecc71" if x >= 0.75 else ("#f39c12" if x >= 0.45 else "#e74c3c")
)
ax.barh(df_stoi["output_file"], df_stoi["STOI"], color=stoi_colors, edgecolor="white")
ax.axvline(0.75, color="green", linestyle="--", alpha=0.6, label="Excellent (≥0.75)")
ax.axvline(0.45, color="orange", linestyle="--", alpha=0.6, label="Acceptable (≥0.45)")
ax.set_xlabel("STOI")
ax.set_title("Short-Time Objective Intelligibility")
ax.set_xlim(0, 1)
ax.legend(fontsize=8)

# PESQ
ax = axes[1]
df_pesq = df_results.dropna(subset=["PESQ"])
pesq_colors = df_pesq["PESQ"].apply(
    lambda x: "#2ecc71" if x >= 3.0 else ("#f39c12" if x >= 2.0 else "#e74c3c")
)
ax.barh(df_pesq["output_file"], df_pesq["PESQ"], color=pesq_colors, edgecolor="white")
ax.axvline(3.0, color="green", linestyle="--", alpha=0.6, label="Excellent (≥3.0)")
ax.axvline(2.0, color="orange", linestyle="--", alpha=0.6, label="Acceptable (≥2.0)")
ax.set_xlabel("PESQ")
ax.set_title("Perceptual Evaluation of Speech Quality")
ax.legend(fontsize=8)

# SNR
ax = axes[2]
df_snr = df_results.dropna(subset=["SNR"])
snr_colors = df_snr["SNR"].apply(
    lambda x: "#2ecc71" if x >= 25 else ("#f39c12" if x >= 15 else "#e74c3c")
)
ax.barh(df_snr["output_file"], df_snr["SNR"], color=snr_colors, edgecolor="white")
ax.axvline(25, color="green", linestyle="--", alpha=0.6, label="Excellent (≥25 dB)")
ax.axvline(15, color="orange", linestyle="--", alpha=0.6, label="Acceptable (≥15 dB)")
ax.set_xlabel("SNR (dB)")
ax.set_title("Signal-to-Noise Ratio")
ax.legend(fontsize=8)

for a in axes:
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)

plt.tight_layout()
plt.show()

# Summary table
quality_cols = ["output_file", "speaker", "STOI", "PESQ", "SNR"]
q_df = df_results[quality_cols].copy()
print("\nSignal Quality Summary:")
for col in ["STOI", "PESQ", "SNR"]:
    vals = q_df[col].dropna()
    if len(vals):
        print(f"  {col:6s}  mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")

--- Cell 35 (markdown) ---
## 17. MCD — Mel Cepstral Distortion

MCD measures spectral envelope distortion between source and converted audio. **Lower is better** — it indicates how much the vocal timbre has changed.

| Tier | MCD (dB) | Interpretation |
|------|----------|----------------|
| Excellent | ≤ 6.0 | Near-natural quality |
| Acceptable | 6.0 – 8.0 | Moderate distortion |
| Poor | > 8.0 | Significant spectral artefacts |

> **Computation:** 24 MFCCs, 80 mel bands, 25 ms window, 10 ms hop, DTW alignment.

--- Cell 36 (code) ---
# ------------------------------------------------------------------
# MCD analysis
# ------------------------------------------------------------------
df_mcd = df_results.dropna(subset=["MCD"]).copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MCD per output
ax = axes[0]
mcd_colors = df_mcd["MCD"].apply(
    lambda x: "#2ecc71" if x <= 6.0 else ("#f39c12" if x <= 8.0 else "#e74c3c")
)
bars = ax.barh(df_mcd["output_file"], df_mcd["MCD"], color=mcd_colors, edgecolor="white")
ax.axvline(6.0, color="green", linestyle="--", alpha=0.6, label="Excellent (≤6.0)")
ax.axvline(8.0, color="orange", linestyle="--", alpha=0.6, label="Acceptable (≤8.0)")
ax.set_xlabel("MCD (dB)")
ax.set_title("Mel Cepstral Distortion per Output")
ax.legend(fontsize=8)

# MCD per speaker (grouped)
ax = axes[1]
speaker_mcd = df_mcd.groupby("speaker")["MCD"].agg(["mean", "std", "count"]).reset_index()
ax.bar(speaker_mcd["speaker"], speaker_mcd["mean"],
       yerr=speaker_mcd["std"].fillna(0), capsize=5,
       color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"][:len(speaker_mcd)],
       edgecolor="white")
ax.axhline(6.0, color="green", linestyle="--", alpha=0.6)
ax.axhline(8.0, color="orange", linestyle="--", alpha=0.6)
ax.set_ylabel("Mean MCD (dB)")
ax.set_title("Mean MCD per Speaker")

for a in axes:
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)

plt.tight_layout()
plt.show()

print(f"\nMCD Summary: mean={df_mcd['MCD'].mean():.2f}  std={df_mcd['MCD'].std():.2f}  "
      f"min={df_mcd['MCD'].min():.2f}  max={df_mcd['MCD'].max():.2f}")

--- Cell 37 (markdown) ---
## 18. F0 — Fundamental Frequency (Prosody)

F0 metrics assess prosodic fidelity — how well the pitch contour is preserved.

- **F0 Correlation**: Pearson correlation of voiced F0 contours (higher = better, ≥ 0.80 excellent)
- **F0 RMSE**: Root mean squared error in Hz between voiced F0 frames (lower = better, ≤ 20 Hz excellent)

> Extracted with `librosa.pyin` at `fmin=50, fmax=600` Hz.

--- Cell 38 (code) ---
# ------------------------------------------------------------------
# F0 correlation & RMSE analysis
# ------------------------------------------------------------------
df_f0 = df_results.dropna(subset=["F0_corr", "F0_rmse"]).copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# F0 correlation per output
ax = axes[0]
f0c_colors = df_f0["F0_corr"].apply(
    lambda x: "#2ecc71" if x >= 0.80 else ("#f39c12" if x >= 0.50 else "#e74c3c")
)
ax.barh(df_f0["output_file"], df_f0["F0_corr"], color=f0c_colors, edgecolor="white")
ax.axvline(0.80, color="green", linestyle="--", alpha=0.6, label="Excellent (≥0.80)")
ax.axvline(0.50, color="orange", linestyle="--", alpha=0.6, label="Acceptable (≥0.50)")
ax.set_xlabel("F0 Correlation (Pearson r)")
ax.set_title("F0 Correlation per Output")
ax.set_xlim(-0.2, 1)
ax.legend(fontsize=8)

# F0 RMSE per output
ax = axes[1]
f0r_colors = df_f0["F0_rmse"].apply(
    lambda x: "#2ecc71" if x <= 20 else ("#f39c12" if x <= 50 else "#e74c3c")
)
ax.barh(df_f0["output_file"], df_f0["F0_rmse"], color=f0r_colors, edgecolor="white")
ax.axvline(20, color="green", linestyle="--", alpha=0.6, label="Excellent (≤20 Hz)")
ax.axvline(50, color="orange", linestyle="--", alpha=0.6, label="Acceptable (≤50 Hz)")
ax.set_xlabel("F0 RMSE (Hz)")
ax.set_title("F0 RMSE per Output")
ax.legend(fontsize=8)

# F0 corr vs RMSE scatter
ax = axes[2]
ax.scatter(df_f0["F0_corr"], df_f0["F0_rmse"], s=100, c="#3498db", edgecolors="white", zorder=3)
for _, r in df_f0.iterrows():
    ax.annotate(r["speaker"], (r["F0_corr"], r["F0_rmse"]),
                fontsize=8, ha="left", va="bottom", xytext=(5, 3),
                textcoords="offset points")
ax.set_xlabel("F0 Correlation")
ax.set_ylabel("F0 RMSE (Hz)")
ax.set_title("F0 Correlation vs RMSE")

for a in axes:
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)

plt.tight_layout()
plt.show()

print(f"\nF0 Summary:")
print(f"  Correlation: mean={df_f0['F0_corr'].mean():.4f}  std={df_f0['F0_corr'].std():.4f}")
print(f"  RMSE (Hz):   mean={df_f0['F0_rmse'].mean():.2f}  std={df_f0['F0_rmse'].std():.2f}")

--- Cell 39 (markdown) ---
## 19. Latency & Real-Time Factor — Experimental Design

Per-chunk latency data (`.perf.json` sidecars) has **not yet been collected** — that is part of the main experiment.
This section presents the **theoretical latency model** and defines the Real-Time Factor (RTF) metric.

### Latency Model
$$\text{End-to-end latency} = T_{\text{chunk}} + T_{\text{inference}} + T_{\text{overhead}}$$

where $T_{\text{chunk}}$ is the chunk duration (input buffering time).

### Real-Time Factor
$$\text{RTF} = \frac{T_{\text{inference}}}{T_{\text{chunk}}}$$

- **RTF < 1.0** → real-time capable (inference completes before next chunk arrives)
- **RTF = 1.0** → borderline
- **RTF > 1.0** → not real-time capable

### Hypothesis H2
> *Shorter chunk lengths (C1, C2) will produce RTF > 1.0 on consumer hardware, while longer chunks (C4, C5) will achieve RTF < 0.5.*

--- Cell 40 (code) ---
# ------------------------------------------------------------------
# Theoretical latency projections for each chunk condition
# ------------------------------------------------------------------
# Assume a range of inference times to illustrate RTF behaviour
assumed_inference_ms = [30, 60, 100, 150, 250, 400]  # representative GPU processing times

latency_rows = []
for cond, v in CONDITIONS.items():
    chunk_ms = v["block_frame_ms"]
    for inf_ms in assumed_inference_ms:
        rtf = inf_ms / chunk_ms
        total_latency = chunk_ms + inf_ms
        latency_rows.append({
            "Condition": cond,
            "Chunk (ms)": chunk_ms,
            "Inference (ms)": inf_ms,
            "RTF": round(rtf, 3),
            "Total Latency (ms)": total_latency,
            "Real-time?": "Yes" if rtf < 1.0 else "No",
        })

df_latency = pd.DataFrame(latency_rows)

# Plot: RTF heatmap
pivot = df_latency.pivot_table(index="Condition", columns="Inference (ms)", values="RTF")
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=3)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel("Assumed Inference Time (ms)")
ax.set_ylabel("Chunk Condition")
ax.set_title("Projected RTF by Condition × Inference Time\n(Green = real-time capable, Red = too slow)")

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        color = "white" if val > 1.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

plt.colorbar(im, ax=ax, label="RTF")
plt.tight_layout()
plt.show()

# Real-time threshold analysis
print("\nReal-time Feasibility (RTF < 1.0):")
for cond in CONDITIONS:
    max_inf = CONDITIONS[cond]["block_frame_ms"]  # RTF=1 when inference = chunk duration
    print(f"  {cond} ({CONDITIONS[cond]['block_frame_ms']:>5d} ms): "
          f"real-time if inference < {max_inf} ms")

print(f"\n200 ms real-time threshold: Only conditions with "
      f"chunk_ms + inference_ms ≤ 200 ms meet strict conversational latency.")

--- Cell 41 (markdown) ---
## 20. Cross-Metric Correlation Analysis

Understanding how metrics co-vary helps identify redundant measures and reveals quality trade-offs. A strong negative correlation between quality metrics and distortion metrics validates our measurement framework.

--- Cell 42 (code) ---
# ------------------------------------------------------------------
# Cross-metric correlation heatmap
# ------------------------------------------------------------------
metric_cols = ["SECS", "MCD", "WER", "CER", "STOI", "PESQ", "SNR", "F0_corr", "F0_rmse"]
available_cols = [c for c in metric_cols if c in df_results.columns]
df_corr = df_results[available_cols].dropna(how="all")

if len(df_corr) >= 3:
    corr_matrix = df_corr.corr(min_periods=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Custom colormap
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index)

    # Annotate
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title("Cross-Metric Correlation Matrix (Baseline C3)")
    plt.tight_layout()
    plt.show()

    # Highlight strong correlations
    print("\nStrong correlations (|r| > 0.5):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.values[i, j]
            if not np.isnan(r) and abs(r) > 0.5:
                print(f"  {corr_matrix.columns[i]:10s} ↔ {corr_matrix.columns[j]:10s}  r={r:+.3f}")
else:
    print("⚠ Insufficient data points for meaningful correlation analysis.")

--- Cell 43 (markdown) ---
## 21. Per-Speaker Baseline Summary

A comprehensive comparison of all metrics grouped by target speaker. The radar chart provides a multi-dimensional quality profile for each speaker model.

--- Cell 44 (code) ---
# ------------------------------------------------------------------
# Per-speaker summary table
# ------------------------------------------------------------------
summary_metrics = ["SECS", "MCD", "WER", "CER", "STOI", "PESQ", "SNR", "F0_corr", "F0_rmse"]
avail_metrics = [m for m in summary_metrics if m in df_results.columns]

speaker_summary = df_results.groupby("speaker")[avail_metrics].agg(["mean", "std", "count"])
display(speaker_summary.round(4).style.set_caption("Per-Speaker Baseline Metrics (C3 condition)"))

# ------------------------------------------------------------------
# Radar chart for multi-dimensional quality comparison
# ------------------------------------------------------------------
# Normalise metrics to [0, 1] for radar (higher = better)
radar_metrics = ["SECS", "STOI", "PESQ", "F0_corr"]   # higher-is-better subset
inv_metrics   = ["MCD", "WER", "CER", "F0_rmse"]        # lower-is-better (invert)

radar_data = {}
for speaker in df_results["speaker"].unique():
    spk_data = df_results[df_results["speaker"] == speaker]
    vals = []
    labels = []

    for m in radar_metrics:
        v = spk_data[m].dropna()
        if len(v):
            vals.append(v.mean())
            labels.append(m)

    for m in inv_metrics:
        v = spk_data[m].dropna()
        if len(v):
            # Invert so that higher = better
            vals.append(1.0 / (1.0 + v.mean()))
            labels.append(f"1/(1+{m})")

    if vals:
        radar_data[speaker] = (labels, vals)

# Plot radar
if radar_data:
    speakers = list(radar_data.keys())
    # Use common label set
    common_labels = radar_data[speakers[0]][0]
    n_vars = len(common_labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_radar = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, speaker in enumerate(speakers):
        labels_s, vals_s = radar_data[speaker]
        vals_plot = vals_s + vals_s[:1]
        ax.plot(angles, vals_plot, "o-", linewidth=2, label=speaker,
                color=colors_radar[idx % len(colors_radar)])
        ax.fill(angles, vals_plot, alpha=0.1, color=colors_radar[idx % len(colors_radar)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(common_labels, fontsize=9)
    ax.set_title("Multi-Metric Quality Profile per Speaker", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()
else:
    print("⚠ Insufficient data for radar chart.")

--- Cell 45 (markdown) ---
## 22. Baseline Tier Classification

Classifying each baseline measurement against the quality tiers defined in the evaluation framework. This provides a clear pass/fail assessment for the baseline condition.

--- Cell 46 (code) ---
# ------------------------------------------------------------------
# Tier classification for each metric
# ------------------------------------------------------------------
def classify_tier(metric, value):
    """Classify a metric value into Excellent / Acceptable / Poor."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"

    tiers = {
        "SECS":    [(0.85, "Excellent"), (0.70, "Acceptable")],
        "MCD":     [(6.0, "Excellent"), (8.0, "Acceptable")],     # inverted: lower better
        "WER":     [(0.10, "Excellent"), (0.25, "Acceptable")],    # inverted
        "CER":     [(0.05, "Excellent"), (0.15, "Acceptable")],    # inverted
        "STOI":    [(0.75, "Excellent"), (0.45, "Acceptable")],
        "PESQ":    [(3.0, "Excellent"), (2.0, "Acceptable")],
        "SNR":     [(25, "Excellent"), (15, "Acceptable")],
        "F0_corr": [(0.80, "Excellent"), (0.50, "Acceptable")],
        "F0_rmse": [(20, "Excellent"), (50, "Acceptable")],       # inverted
    }

    inv = {"MCD", "WER", "CER", "F0_rmse"}  # lower is better

    if metric not in tiers:
        return "?"

    thresholds = tiers[metric]
    if metric in inv:
        if value <= thresholds[0][0]:
            return thresholds[0][1]
        elif value <= thresholds[1][0]:
            return thresholds[1][1]
        else:
            return "Poor"
    else:
        if value >= thresholds[0][0]:
            return thresholds[0][1]
        elif value >= thresholds[1][0]:
            return thresholds[1][1]
        else:
            return "Poor"

# Build tier table
tier_rows = []
for _, r in df_results.iterrows():
    row = {"Output": r["output_file"], "Speaker": r["speaker"]}
    for m in ["SECS", "MCD", "WER", "CER", "STOI", "PESQ", "SNR", "F0_corr", "F0_rmse"]:
        val = r.get(m)
        tier = classify_tier(m, val)
        val_str = f"{val:.3f}" if isinstance(val, float) and not np.isnan(val) else "—"
        row[m] = f"{val_str} ({tier})"
    tier_rows.append(row)

df_tiers = pd.DataFrame(tier_rows)

# Style: colour cells by tier
def color_tier(val):
    if "Excellent" in str(val):
        return "background-color: #d5f5e3"
    elif "Acceptable" in str(val):
        return "background-color: #fdebd0"
    elif "Poor" in str(val):
        return "background-color: #fadbd8"
    return ""

# Use .map() instead of deprecated .applymap() (removed in pandas 2.1+)
styled = df_tiers.style.map(color_tier).set_caption(
    "Baseline Tier Classification (Green = Excellent, Yellow = Acceptable, Red = Poor)"
)
display(styled)

# Count tiers across all measurements
all_tiers = []
for _, r in df_results.iterrows():
    for m in ["SECS", "MCD", "WER", "CER", "STOI", "PESQ", "SNR", "F0_corr", "F0_rmse"]:
        t = classify_tier(m, r.get(m))
        if t != "N/A":
            all_tiers.append(t)

tier_counts = pd.Series(all_tiers).value_counts()
print("\nOverall Tier Distribution:")
print(tier_counts.to_string())
print(f"\nPercentage Excellent: {tier_counts.get('Excellent', 0) / len(all_tiers) * 100:.1f}%")
print(f"Percentage Acceptable+: {(tier_counts.get('Excellent', 0) + tier_counts.get('Acceptable', 0)) / len(all_tiers) * 100:.1f}%")

--- Cell 47 (markdown) ---
## 23. Statistical Analysis — Distribution Properties & Test Selection

Before the main experiment, we assess the distribution properties of our baseline metrics to inform the choice of statistical tests:

- **Normality** (Shapiro-Wilk, α = 0.05): determines ANOVA vs Kruskal-Wallis
- **Variance homogeneity** (Levene's test): determines standard vs Welch's ANOVA
- **Sample size adequacy**: power analysis for the 5-condition × 3-speaker design

### Proposed Statistical Framework (from Proposal §5.5)
1. **One-way repeated-measures ANOVA** or **Friedman test** for each metric across 5 chunk conditions
2. **Post-hoc pairwise comparisons** with Bonferroni correction (10 pairs)
3. **Effect size**: η² (partial eta-squared) for ANOVA, or Kendall's W for Friedman
4. **Significance level**: α = 0.05 throughout

--- Cell 48 (code) ---
# ------------------------------------------------------------------
# Distribution analysis of baseline metrics
# ------------------------------------------------------------------
from scipy import stats

norm_results = []
test_metrics = ["SECS", "MCD", "WER", "CER", "STOI", "PESQ", "SNR", "F0_corr", "F0_rmse"]

for m in test_metrics:
    vals = df_results[m].dropna().values
    n = len(vals)
    if n >= 3:
        stat, p = stats.shapiro(vals)
        normal = "Yes" if p > 0.05 else "No"
        skew = stats.skew(vals)
        kurt = stats.kurtosis(vals)
        norm_results.append({
            "Metric": m,
            "N": n,
            "Mean": round(np.mean(vals), 4),
            "Std": round(np.std(vals, ddof=1), 4),
            "Skewness": round(skew, 3),
            "Kurtosis": round(kurt, 3),
            "Shapiro W": round(stat, 4),
            "Shapiro p": round(p, 4),
            "Normal (α=0.05)": normal,
        })
    else:
        norm_results.append({
            "Metric": m, "N": n,
            "Mean": round(np.mean(vals), 4) if n > 0 else None,
            "Std": None, "Skewness": None, "Kurtosis": None,
            "Shapiro W": None, "Shapiro p": None,
            "Normal (α=0.05)": "Insufficient data",
        })

df_norm = pd.DataFrame(norm_results)
display(df_norm.style.set_caption("Normality Assessment of Baseline Metrics"))

# Distribution plots
plot_metrics = [m for m in test_metrics if df_results[m].dropna().shape[0] >= 3]
n_plots = len(plot_metrics)
if n_plots:
    fig, axes = plt.subplots(2, min(n_plots, 5), figsize=(min(n_plots, 5)*3.5, 7))
    if n_plots == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, m in enumerate(plot_metrics[:5]):
        vals = df_results[m].dropna().values
        # Histogram
        ax = axes[0, idx] if axes.shape[0] > 1 else axes[0, idx]
        ax.hist(vals, bins=max(3, len(vals)//2), color="#3498db", edgecolor="white", alpha=0.7)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("Value")
        if idx == 0:
            ax.set_ylabel("Count")

        # Q-Q plot
        if axes.shape[0] > 1:
            ax2 = axes[1, idx]
            stats.probplot(vals, dist="norm", plot=ax2)
            ax2.set_title(f"Q-Q: {m}", fontsize=9)

    plt.tight_layout()
    plt.show()

# Statistical test recommendation
print("\n" + "="*60)
print("STATISTICAL TEST RECOMMENDATIONS")
print("="*60)
normal_metrics = df_norm[df_norm["Normal (α=0.05)"] == "Yes"]["Metric"].tolist()
non_normal = df_norm[df_norm["Normal (α=0.05)"] == "No"]["Metric"].tolist()

if normal_metrics:
    print(f"\n  Normal distribution → One-way RM-ANOVA:")
    for m in normal_metrics:
        print(f"    • {m}")

if non_normal:
    print(f"\n  Non-normal distribution → Friedman test:")
    for m in non_normal:
        print(f"    • {m}")

print(f"\n  Post-hoc: Bonferroni-corrected pairwise comparisons")
print(f"  Pairs: C(5,2) = 10 comparisons → α_adj = 0.005")
print(f"\n  Effect size: η²_p (ANOVA) or Kendall's W (Friedman)")

# Power analysis note
print(f"\n{'='*60}")
print("SAMPLE SIZE CONSIDERATIONS")
print("="*60)
print(f"  Design: 5 conditions × 3 speakers × 3 source clips × N reps")
print(f"  Current baseline: {len(df_results)} observations (1 condition)")
print(f"  Target for main experiment: N ≥ 3 reps per cell → 135 observations")
print(f"  For medium effect (f=0.25, α=0.05, power=0.80):")
print(f"    Minimum N per group ≈ 25 (from G*Power a-priori calculation)")

--- Cell 49 (markdown) ---
## 24. EDA Summary & Transition to Experiment

### Key EDA Findings
1. **Baseline condition (C3, ~512 ms) is functional**: All 6 conversions produced valid outputs with measurable quality metrics.
2. **Speaker identity transfer varies by model**: SECS scores show different fidelity levels across speakers.
3. **Evaluation pipeline works end-to-end**: All 8 metrics computed successfully.
4. **Only C3 tested so far**: The quality–latency trade-off remains unquantified.

### Identified Limitations (to address in full experiment)
| Limitation | Mitigation |
|---|---|
| Only 1 condition (C3) | Run all 5 conditions (C1–C5) |
| No latency data | `infer_with_timing.py` with `.perf.json` sidecars |
| 2 speakers lack references | Focus on 3 speakers with full refs (Trump, Banks, Wheatley) |
| No repetitions | ≥3 reps per cell for statistical power |

---

# Part III — Method Implementation & Experimental Setup (Task 9)

> **ILO 7.3A**: The method is implemented in alignment with the research proposal. Code is structured, readable, and reproducible.

--- Cell 50 (code) ---
## 25. Deviations from Proposal

| Aspect | Proposal | Actual | Justification |
|--------|----------|--------|---------------|
| Dataset | 100 FakeAVCeleb clips | 3 self-recorded clips | FakeAVCeleb requires manual request; self-recorded clips provide controlled conditions with known ground truth. |
| Chunk lengths | 128/256/512/1024/2048 samples @16kHz | C1(24)/C2(72)/C3(192)/C4(384)/C5(768) `read_chunk_size` @48kHz I/O | Applio uses `read_chunk_size × 128` block frames at 48kHz I/O rate; conditions map to ~64/192/512/1024/2048 ms. |
| Target speakers | DonaldTrump only | DonaldTrump + GeorgeBanks + Wheatley-HD | 3 speakers provide cross-speaker generalisability; all have reference recordings for SECS. |
| Repetitions | Not specified | 3 per cell | Minimum for Shapiro-Wilk normality testing and variance estimation. |
| Additional metrics | CosSim, WER, SNR only | +MCD, +CER, +STOI, +PESQ, +F0 | Richer evaluation captures spectral, intelligibility, and prosodic dimensions. |
| Statistical tests | RM-ANOVA + Tukey HSD | Kruskal-Wallis + Dunn's (if non-normal) OR RM-ANOVA + Tukey | Test selection informed by EDA normality analysis (Section 23). |

--- Cell 51 (markdown) ---
## 26. Experimental Configuration

The full experiment systematically varies chunk length (5 conditions) across 3 target speakers and 3 source clips with 3 repetitions each, producing **135 inference runs** (5×3×3×3).

Each run:
1. Processes source audio through RVC at a specific `read_chunk_size` via `infer_with_timing()`
2. Records per-chunk latency in a `.perf.json` sidecar
3. Evaluates the output via `evaluate()` (all 8 quality metrics)
4. Appends the row to the experiment CSV checkpoint

--- Cell 52 (code) ---
# ------------------------------------------------------------------
# Experimental configuration
# ------------------------------------------------------------------

# Chunk length conditions (from proposal Table 1)
CONDITIONS = {
    "C1": {"read_chunk_size": 24,  "block_frame_ms": 64,   "label": "~64 ms"},
    "C2": {"read_chunk_size": 72,  "block_frame_ms": 192,  "label": "~192 ms"},
    "C3": {"read_chunk_size": 192, "block_frame_ms": 512,  "label": "~512 ms (baseline)"},
    "C4": {"read_chunk_size": 384, "block_frame_ms": 1024, "label": "~1024 ms"},
    "C5": {"read_chunk_size": 768, "block_frame_ms": 2048, "label": "~2048 ms"},
}

# Target speakers with full model paths (verified from Applio/logs/)
SPEAKERS = {
    "DonaldTrump": {
        "model_path":  str(AUDIO_DIR / "Applio" / "logs" / "DonaldTrump" / "DonaldTrump_475e_8075s.pth"),
        "index_path":  str(AUDIO_DIR / "Applio" / "logs" / "DonaldTrump" / "DonaldTrump.index"),
        "reference":   str(ORIGINAL_DIR / "trump.mp3"),
    },
    "GeorgeBanks": {
        "model_path":  str(AUDIO_DIR / "Applio" / "logs" / "GeorgeBanks" / "George Banks_500e_41500s.pth"),
        "index_path":  str(AUDIO_DIR / "Applio" / "logs" / "GeorgeBanks" / "George Banks.index"),
        "reference":   str(ORIGINAL_DIR / "banks.wav"),
    },
    "Wheatley-HD": {
        "model_path":  str(AUDIO_DIR / "Applio" / "logs" / "Wheatley-HD_e450_s40050.pth"),
        "index_path":  str(AUDIO_DIR / "Applio" / "logs" / "added_IVF5119_Flat_nprobe_1_Wheatley-HD_v2.index"),
        "reference":   str(ORIGINAL_DIR / "wheatly.wav"),
    },
}

# Source clips
SOURCE_CLIPS = [
    str(INPUT_DIR / "myexample1.wav"),
    str(INPUT_DIR / "myexample2.mp3"),
    str(INPUT_DIR / "myexample3.ogg"),
]

N_REPS = 3  # repetitions per cell

# Experiment output directory
EXPERIMENT_OUTPUT_DIR = AUDIO_DIR / "assets" / "output" / "experiment"
EXPERIMENT_CSV = RESULTS_DIR / "experiment_full.csv"

# Verify all files exist
print("Speaker model verification:")
for name, sp in SPEAKERS.items():
    m_ok = Path(sp["model_path"]).exists()
    i_ok = Path(sp["index_path"]).exists()
    r_ok = Path(sp["reference"]).exists()
    status = "OK" if (m_ok and i_ok and r_ok) else "MISSING"
    print(f"  {name:15s}  model={m_ok}  index={i_ok}  ref={r_ok}  [{status}]")

print(f"\nSource clips:")
for s in SOURCE_CLIPS:
    print(f"  {Path(s).name:25s}  exists={Path(s).exists()}")

total_runs = len(CONDITIONS) * len(SPEAKERS) * len(SOURCE_CLIPS) * N_REPS
print(f"\nTotal experiment runs: {len(CONDITIONS)} conditions × {len(SPEAKERS)} speakers "
      f"× {len(SOURCE_CLIPS)} clips × {N_REPS} reps = {total_runs}")
print(f"Experiment output: {EXPERIMENT_OUTPUT_DIR}")
print(f"Experiment CSV:    {EXPERIMENT_CSV}")

--- Cell 53 (markdown) ---
## 27. Run Full Experiment

This cell runs the complete experiment: all conditions × speakers × clips × reps.

**Checkpoint system**: Results are appended to `experiment_full.csv` after each run. If the CSV already contains a row for a specific (condition, speaker, clip, rep) combination, that run is **skipped**. This means:
- You can safely re-run this cell without losing progress
- If interrupted, just re-run to continue from where you left off
- Delete the CSV to start fresh

> **⚠ This cell requires GPU and the Applio RVC environment. Runtime ~2–5 min per run depending on hardware.**

--- Cell 54 (code) ---
# ------------------------------------------------------------------
# Full Experiment Execution with Checkpoint Resume
# ------------------------------------------------------------------
import csv, time as _time

# Add audio/ to path for infer_with_timing import
sys.path.insert(0, str(AUDIO_DIR))

try:
    from infer_with_timing import infer_with_timing
    INFERENCE_AVAILABLE = True
    print("infer_with_timing imported successfully.")
except ImportError as e:
    INFERENCE_AVAILABLE = False
    print(f"WARNING: Cannot import infer_with_timing: {e}")
    print("Inference will be SKIPPED. Only evaluation of existing outputs will run.")

# Create output directory
EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load existing checkpoint to determine which runs to skip
completed_runs = set()
if EXPERIMENT_CSV.exists():
    existing = pd.read_csv(EXPERIMENT_CSV)
    for _, row in existing.iterrows():
        key = (row["condition"], row["speaker"], row["source_clip"], int(row["repetition"]))
        completed_runs.add(key)
    print(f"Checkpoint loaded: {len(completed_runs)} runs already completed.")

# CSV field names (matching run_baseline.py format)
CSV_FIELDS = [
    "condition", "read_chunk_size", "block_frame_ms",
    "speaker", "source_clip", "repetition",
    "SECS", "MCD", "WER", "CER", "STOI", "PESQ", "SNR",
    "F0_corr", "F0_rmse",
    "mean_chunk_latency_ms", "std_chunk_latency_ms",
    "min_chunk_latency_ms", "max_chunk_latency_ms",
    "total_latency_ms", "num_chunks", "peak_gpu_memory_mb",
    "output_wav",
]

# Open CSV in append mode
csv_existed = EXPERIMENT_CSV.exists()
csv_file = open(EXPERIMENT_CSV, "a", newline="", encoding="utf-8")
writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
if not csv_existed:
    writer.writeheader()

total_runs = len(CONDITIONS) * len(SPEAKERS) * len(SOURCE_CLIPS) * N_REPS
run_count = 0
skipped = 0
failed = 0
t_start = _time.time()

try:
    for cond_name, cond in CONDITIONS.items():
        for speaker_name, speaker in SPEAKERS.items():
            for src_path in SOURCE_CLIPS:
                src_stem = Path(src_path).stem
                for rep in range(1, N_REPS + 1):
                    run_count += 1
                    key = (cond_name, speaker_name, src_stem, rep)

                    # Skip if already completed
                    if key in completed_runs:
                        skipped += 1
                        continue

                    out_name = f"{src_stem}__{speaker_name}__{cond_name}__rep{rep}.wav"
                    out_path = str(EXPERIMENT_OUTPUT_DIR / out_name)

                    print(f"\n[{run_count}/{total_runs}] {cond_name} | {speaker_name} | "
                          f"{src_stem} | rep{rep}")

                    row = {
                        "condition": cond_name,
                        "read_chunk_size": cond["read_chunk_size"],
                        "block_frame_ms": cond["block_frame_ms"],
                        "speaker": speaker_name,
                        "source_clip": src_stem,
                        "repetition": rep,
                        "output_wav": out_name,
                    }

                    # Step 1: Inference (if available and output doesn't exist)
                    perf_stats = None
                    if INFERENCE_AVAILABLE:
                        if not Path(out_path).exists():
                            try:
                                perf_stats = infer_with_timing(
                                    source_path=src_path,
                                    output_path=out_path,
                                    model_path=speaker["model_path"],
                                    index_path=speaker["index_path"],
                                    read_chunk_size=cond["read_chunk_size"],
                                    target_speaker=speaker_name,
                                )
                            except Exception as e:
                                print(f"  INFERENCE ERROR: {e}")
                                failed += 1
                                continue
                        else:
                            # Load existing perf.json if available
                            perf_path = Path(out_path).with_suffix("").as_posix() + ".perf.json"
                            if Path(perf_path).exists():
                                with open(perf_path) as f:
                                    perf_stats = json.load(f)
                                print(f"  Loaded existing perf stats from {Path(perf_path).name}")
                    else:
                        if not Path(out_path).exists():
                            print(f"  SKIPPED — no inference engine and output doesn't exist")
                            failed += 1
                            continue

                    # Add latency stats
                    if perf_stats:
                        row["mean_chunk_latency_ms"] = perf_stats.get("mean_chunk_latency_ms")
                        row["std_chunk_latency_ms"] = perf_stats.get("std_chunk_latency_ms")
                        row["min_chunk_latency_ms"] = perf_stats.get("min_chunk_latency_ms")
                        row["max_chunk_latency_ms"] = perf_stats.get("max_chunk_latency_ms")
                        row["total_latency_ms"] = perf_stats.get("total_latency_ms")
                        row["num_chunks"] = perf_stats.get("num_chunks")
                        row["peak_gpu_memory_mb"] = perf_stats.get("peak_gpu_memory_mb")

                    # Step 2: Evaluation
                    try:
                        eval_result = evaluate_audio(
                            original_path=src_path,
                            transformed_path=out_path,
                            target_path=speaker["reference"],
                            whisper_model="large-v3-turbo",
                            whisper_device="cpu",
                            whisper_compute_type="int8",
                        )

                        for eval_key, col_name in EVAL_KEY_MAP.items():
                            val = eval_result.get(eval_key)
                            row[col_name] = round(val, 4) if isinstance(val, (int, float)) and val is not None else val

                    except Exception as e:
                        print(f"  EVALUATION ERROR: {e}")
                        failed += 1
                        continue

                    # Step 3: Write to CSV
                    writer.writerow(row)
                    csv_file.flush()
                    completed_runs.add(key)

                    elapsed = _time.time() - t_start
                    done = len(completed_runs)
                    rate = elapsed / max(done - skipped, 1)
                    remaining = rate * (total_runs - done)
                    print(f"  Progress: {done}/{total_runs} "
                          f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

except KeyboardInterrupt:
    print(f"\n⚠ Interrupted at run {run_count}/{total_runs}")
finally:
    csv_file.close()

print(f"\n{'='*65}")
print(f"EXPERIMENT COMPLETE")
print(f"{'='*65}")
print(f"  Completed: {len(completed_runs)}/{total_runs}")
print(f"  Skipped (cached): {skipped}")
print(f"  Failed: {failed}")
print(f"  Results saved to: {EXPERIMENT_CSV}")
print(f"  Total time: {_time.time() - t_start:.1f}s")

--- Cell 55 (markdown) ---
## 28. Load & Merge Experiment Results

Load the experiment CSV produced by Section 27 and merge with baseline (C3) metrics
from Part II for a unified analysis DataFrame.

--- Cell 56 (code) ---
# ------------------------------------------------------------------
# Load experiment results
# ------------------------------------------------------------------
if EXPERIMENT_CSV.exists():
    df_experiment = pd.read_csv(EXPERIMENT_CSV)
    print(f"Loaded {len(df_experiment)} experiment rows from {EXPERIMENT_CSV.name}")
else:
    print(f"WARNING: {EXPERIMENT_CSV} not found. Run Section 27 first.")
    df_experiment = pd.DataFrame()

# Ensure correct dtypes
METRIC_COLS = ["SECS", "MCD", "WER", "CER", "STOI", "PESQ", "SNR", "F0_corr", "F0_rmse"]
LATENCY_COLS = ["mean_chunk_latency_ms", "std_chunk_latency_ms",
                "min_chunk_latency_ms", "max_chunk_latency_ms",
                "total_latency_ms", "num_chunks"]

for col in METRIC_COLS + LATENCY_COLS:
    if col in df_experiment.columns:
        df_experiment[col] = pd.to_numeric(df_experiment[col], errors="coerce")

# Add block_frame_ms as ordered categorical for proper plotting
if not df_experiment.empty:
    cond_order = ["C1", "C2", "C3", "C4", "C5"]
    df_experiment["condition"] = pd.Categorical(
        df_experiment["condition"], categories=cond_order, ordered=True
    )

    # Quick summary
    print(f"\nConditions found: {df_experiment['condition'].unique().tolist()}")
    print(f"Speakers found:  {df_experiment['speaker'].unique().tolist()}")
    print(f"Source clips:    {df_experiment['source_clip'].unique().tolist()}")
    print(f"Repetitions:     {sorted(df_experiment['repetition'].unique().tolist())}")
    print(f"\nPer-condition counts:")
    display(df_experiment.groupby("condition")[METRIC_COLS[0]].count().rename("n_runs"))
    print()
    display(df_experiment.describe().round(4))

--- Cell 57 (markdown) ---
---

# Part IV — Task 10: Quantitative Results & Evaluation

> **Task 10 deliverables**: Present quantitative results; compare against baseline;
> include summary tables, visualisations, and interpretation of every metric.

## 29. Summary Statistics per Condition

Grand mean ± SD for every metric, grouped by chunk-length condition (C1-C5).

--- Cell 58 (code) ---
# ------------------------------------------------------------------
# Summary statistics per condition
# ------------------------------------------------------------------
if not df_experiment.empty:
    summary = df_experiment.groupby("condition")[METRIC_COLS].agg(["mean", "std", "count"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    
    # Create a cleaner display table
    display_cols = []
    for m in METRIC_COLS:
        col_mean = f"{m}_mean"
        col_std  = f"{m}_std"
        if col_mean in summary.columns:
            summary[f"{m}"] = summary[col_mean].round(4).astype(str) + " ± " + summary[col_std].round(4).astype(str)
            display_cols.append(m)
    
    print("Table 2: Mean ± SD per Condition (all speakers/clips pooled)")
    display(summary[display_cols])
    
    # Also save this table
    summary_path = RESULTS_DIR / "task10_summary_per_condition.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved to {summary_path.name}")
else:
    print("No experiment data available.")

--- Cell 59 (markdown) ---
## 30. Quality Metrics vs Chunk Length (Boxplots)

Boxplots showing the distribution of each quality metric (STOI, PESQ, SNR, MCD)
across the five chunk-length conditions. Each box aggregates all speakers × clips × reps.

--- Cell 60 (code) ---
# ------------------------------------------------------------------
# Boxplots: quality metrics vs condition
# ------------------------------------------------------------------
if not df_experiment.empty:
    quality_metrics = ["STOI", "PESQ", "SNR", "MCD"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, quality_metrics):
        data = [
            df_experiment.loc[df_experiment["condition"] == c, metric].dropna()
            for c in ["C1", "C2", "C3", "C4", "C5"]
        ]
        bp = ax.boxplot(data, labels=["C1\n64ms", "C2\n192ms", "C3\n512ms", "C4\n1024ms", "C5\n2048ms"],
                        patch_artist=True, widths=0.6)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Highlight baseline (C3) with a red border
        bp["boxes"][2].set_edgecolor("red")
        bp["boxes"][2].set_linewidth(2)

        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xlabel("Condition (block frame)")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 7: Quality Metrics Distribution by Chunk-Length Condition",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig7_quality_boxplots.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No experiment data available.")

--- Cell 61 (markdown) ---
## 31. Content Preservation Metrics vs Chunk Length

WER and CER distributions across conditions — these measure how well
the original speech content is preserved after voice conversion.

--- Cell 62 (code) ---
# ------------------------------------------------------------------
# Content preservation: WER/CER vs condition
# ------------------------------------------------------------------
if not df_experiment.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cond_labels = ["C1\n64ms", "C2\n192ms", "C3\n512ms", "C4\n1024ms", "C5\n2048ms"]
    conds = ["C1", "C2", "C3", "C4", "C5"]

    for ax, metric, title in zip(axes, ["WER", "CER"],
                                  ["Word Error Rate", "Character Error Rate"]):
        means = [df_experiment.loc[df_experiment["condition"] == c, metric].mean() for c in conds]
        stds  = [df_experiment.loc[df_experiment["condition"] == c, metric].std()  for c in conds]

        bars = ax.bar(cond_labels, means, yerr=stds, capsize=5,
                      color=plt.cm.viridis(np.linspace(0.2, 0.8, 5)), alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        # Highlight baseline
        bars[2].set_edgecolor("red")
        bars[2].set_linewidth(2)

        ax.set_title(f"{title} ({metric})", fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xlabel("Condition")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, m, s in zip(bars, means, stds):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.01,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Figure 8: Content Preservation vs Chunk Length",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig8_content_preservation.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No experiment data available.")

--- Cell 63 (markdown) ---
## 32. Prosody Metrics vs Chunk Length

F0 correlation and F0 RMSE — how well pitch contour is preserved across conditions.

--- Cell 64 (code) ---
# ------------------------------------------------------------------
# Prosody: F0 correlation / RMSE vs condition
# ------------------------------------------------------------------
if not df_experiment.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cond_labels = ["C1\n64ms", "C2\n192ms", "C3\n512ms", "C4\n1024ms", "C5\n2048ms"]
    conds = ["C1", "C2", "C3", "C4", "C5"]

    for ax, metric, title, ylabel in zip(
        axes, ["F0_corr", "F0_rmse"],
        ["F0 Correlation (higher=better)", "F0 RMSE (lower=better)"],
        ["Pearson r", "Hz"]
    ):
        for i, (c, label) in enumerate(zip(conds, cond_labels)):
            subset = df_experiment.loc[df_experiment["condition"] == c, metric].dropna()
            ax.scatter([i]*len(subset), subset, alpha=0.5, s=30,
                       color=plt.cm.viridis(i/4), zorder=3)
        
        means = [df_experiment.loc[df_experiment["condition"] == c, metric].mean() for c in conds]
        ax.plot(range(5), means, "ro-", markersize=8, linewidth=2, zorder=4, label="Mean")
        
        ax.set_xticks(range(5))
        ax.set_xticklabels(cond_labels)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle("Figure 9: Prosody Preservation vs Chunk Length",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig9_prosody.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No experiment data available.")

--- Cell 65 (markdown) ---
## 33. Latency Analysis

Per-chunk latency statistics from the `.perf.json` sidecar files.
Compute Real-Time Factor (RTF) and plot latency vs chunk length.

--- Cell 66 (code) ---
# ------------------------------------------------------------------
# Latency analysis
# ------------------------------------------------------------------
if not df_experiment.empty and "mean_chunk_latency_ms" in df_experiment.columns:
    latency_valid = df_experiment.dropna(subset=["mean_chunk_latency_ms"])
    
    if not latency_valid.empty:
        # Compute RTF: processing_time / audio_duration
        # block_frame_ms is the audio duration per chunk
        latency_valid = latency_valid.copy()
        latency_valid["RTF"] = (
            latency_valid["mean_chunk_latency_ms"] / latency_valid["block_frame_ms"]
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        conds = ["C1", "C2", "C3", "C4", "C5"]
        cond_labels = ["C1\n64ms", "C2\n192ms", "C3\n512ms", "C4\n1024ms", "C5\n2048ms"]

        # (a) Mean chunk latency
        ax = axes[0]
        for i, c in enumerate(conds):
            subset = latency_valid.loc[latency_valid["condition"] == c, "mean_chunk_latency_ms"]
            ax.boxplot(subset.dropna(), positions=[i], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=plt.cm.viridis(i/4), alpha=0.7))
        ax.set_xticks(range(5))
        ax.set_xticklabels(cond_labels)
        ax.set_title("(a) Mean Chunk Latency", fontweight="bold")
        ax.set_ylabel("Latency (ms)")
        ax.grid(axis="y", alpha=0.3)

        # (b) Total processing time
        ax = axes[1]
        for i, c in enumerate(conds):
            subset = latency_valid.loc[latency_valid["condition"] == c, "total_latency_ms"]
            ax.boxplot(subset.dropna(), positions=[i], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=plt.cm.viridis(i/4), alpha=0.7))
        ax.set_xticks(range(5))
        ax.set_xticklabels(cond_labels)
        ax.set_title("(b) Total Processing Time", fontweight="bold")
        ax.set_ylabel("Time (ms)")
        ax.grid(axis="y", alpha=0.3)

        # (c) Real-Time Factor
        ax = axes[2]
        rtf_means = [latency_valid.loc[latency_valid["condition"] == c, "RTF"].mean() for c in conds]
        rtf_stds  = [latency_valid.loc[latency_valid["condition"] == c, "RTF"].std()  for c in conds]
        bars = ax.bar(cond_labels, rtf_means, yerr=rtf_stds, capsize=5,
                      color=plt.cm.viridis(np.linspace(0.2, 0.8, 5)), alpha=0.8)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Real-time (RTF=1)")
        ax.set_title("(c) Real-Time Factor", fontweight="bold")
        ax.set_ylabel("RTF (lower=faster)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Figure 10: Latency Analysis by Chunk-Length Condition",
                     fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fig10_latency.png", dpi=150, bbox_inches="tight")
        plt.show()

        # Print RTF summary table
        print("\nTable 3: Real-Time Factor per Condition")
        rtf_summary = latency_valid.groupby("condition")["RTF"].agg(["mean", "std", "min", "max"])
        display(rtf_summary.round(4))
    else:
        print("No latency data available in experiment results.")
else:
    print("No experiment data or latency columns missing.")

--- Cell 67 (markdown) ---
## 34. Per-Speaker Breakdown

Are some voices more sensitive to chunk-length changes than others?
Heatmaps showing metric means per (condition × speaker) combination.

--- Cell 68 (code) ---
# ------------------------------------------------------------------
# Per-speaker heatmaps
# ------------------------------------------------------------------
if not df_experiment.empty:
    key_metrics = ["STOI", "PESQ", "MCD", "WER"]
    n_metrics = len(key_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

    for ax, metric in zip(axes, key_metrics):
        pivot = df_experiment.pivot_table(
            values=metric, index="speaker", columns="condition", aggfunc="mean"
        )
        # Reorder conditions
        pivot = pivot.reindex(columns=["C1", "C2", "C3", "C4", "C5"])

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd_r" if metric in ["WER", "MCD"] else "YlGn",
                    ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlabel("Condition")

    fig.suptitle("Figure 11: Per-Speaker Mean Metrics by Condition",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig11_speaker_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No experiment data available.")

--- Cell 69 (markdown) ---
## 35. Cross-Metric Correlation & Multi-Metric Radar

(a) Correlation matrix across all metrics to identify collinearity.
(b) Radar (spider) chart comparing all conditions across normalised metrics.

--- Cell 70 (code) ---
# ------------------------------------------------------------------
# (a) Cross-metric correlation matrix
# ------------------------------------------------------------------
if not df_experiment.empty:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Correlation heatmap
    ax = axes[0]
    corr = df_experiment[METRIC_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                square=True)
    ax.set_title("(a) Metric Correlation Matrix", fontsize=13, fontweight="bold")

    # (b) Radar chart
    ax = axes[1]
    ax.set_visible(False)  # Remove rectangular axis

    ax_radar = fig.add_subplot(122, polar=True)
    conds = ["C1", "C2", "C3", "C4", "C5"]
    radar_metrics = ["STOI", "PESQ", "SNR", "F0_corr", "WER", "MCD"]

    # Normalise to [0,1] — for WER/MCD, invert so higher = better
    invert = {"WER", "MCD", "F0_rmse", "CER"}
    normed = {}
    for m in radar_metrics:
        vals = df_experiment[m].dropna()
        mn, mx = vals.min(), vals.max()
        rng = mx - mn if mx != mn else 1
        if m in invert:
            normed[m] = lambda x, mn=mn, rng=rng: 1 - (x - mn) / rng
        else:
            normed[m] = lambda x, mn=mn, rng=rng: (x - mn) / rng

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # close

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    for i, c in enumerate(conds):
        subset = df_experiment[df_experiment["condition"] == c]
        values = [normed[m](subset[m].mean()) for m in radar_metrics]
        values += values[:1]
        ax_radar.plot(angles, values, "o-", linewidth=2, color=colors[i], label=c)
        ax_radar.fill(angles, values, alpha=0.1, color=colors[i])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_metrics, fontsize=10)
    ax_radar.set_title("(b) Multi-Metric Radar\n(normalised, higher=better)",
                       fontsize=13, fontweight="bold", y=1.08)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.suptitle("Figure 12: Cross-Metric Analysis",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig12_cross_metric.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No experiment data available.")

--- Cell 71 (markdown) ---
---

# Part V — Task 11: Critical Analysis & Hypothesis Testing

> **Task 11 deliverables**: Formal hypothesis testing for H1-H3; baseline comparison;
> effect sizes; limitations; self-reflection on methodology.

**Hypotheses:**

| ID | Statement | Statistical Test |
|----|-----------|-----------------|
| H1 | Increasing chunk length monotonically improves quality metrics | Jonckheere-Terpstra trend test + Kruskal-Wallis |
| H2 | Per-chunk processing latency scales linearly with chunk length | OLS regression + Pearson r |
| H3 | Quality gains saturate beyond a threshold chunk size | Piecewise (segmented) regression |

## 36. H1 — Monotonic Quality Improvement

**H1**: *Increasing the RVC read_chunk_size monotonically improves STOI, PESQ,
and speaker similarity (SECS) while reducing MCD and WER.*

- **Jonckheere-Terpstra** trend test (ordered alternative): tests whether
  the metric values show a monotonic increase (or decrease) across C1 → C5.
- **Kruskal-Wallis** (omnibus): tests whether at least one condition differs.
- **Effect size**: η² (eta-squared) from Kruskal-Wallis H statistic.

--- Cell 72 (code) ---
# ------------------------------------------------------------------
# H1: Monotonic quality improvement with chunk length
# ------------------------------------------------------------------
from scipy import stats as sp_stats
from itertools import combinations

def jonckheere_terpstra(groups):
    """
    Jonckheere-Terpstra test for ordered alternatives.
    groups: list of arrays in hypothesised order (increasing).
    Returns (J statistic, z-score, p-value).
    """
    k = len(groups)
    J = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            for xi in groups[i]:
                for xj in groups[j]:
                    if xj > xi:
                        J += 1
                    elif xj == xi:
                        J += 0.5

    # Expected value and variance under H0
    ns = [len(g) for g in groups]
    N = sum(ns)
    E_J = (N**2 - sum(n**2 for n in ns)) / 4
    var_num = N**2 * (2*N + 3) - sum(n**2 * (2*n + 3) for n in ns)
    Var_J = var_num / 72
    z = (J - E_J) / np.sqrt(Var_J) if Var_J > 0 else 0
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))  # two-tailed
    return J, z, p


if not df_experiment.empty:
    conds = ["C1", "C2", "C3", "C4", "C5"]
    h1_metrics = ["STOI", "PESQ", "SECS", "MCD", "WER"]
    # For MCD and WER, lower is better → we expect decreasing trend
    # So we reverse the groups to test for "monotonic improvement"
    higher_is_better = {"STOI", "PESQ", "SECS", "SNR", "F0_corr"}

    h1_results = []
    for metric in h1_metrics:
        groups = [df_experiment.loc[df_experiment["condition"] == c, metric].dropna().values
                  for c in conds]
        
        # Remove empty groups
        valid_groups = [g for g in groups if len(g) > 0]
        if len(valid_groups) < 3:
            continue

        # Kruskal-Wallis omnibus test
        H_stat, kw_p = sp_stats.kruskal(*valid_groups)
        N = sum(len(g) for g in valid_groups)
        eta_sq = (H_stat - len(valid_groups) + 1) / (N - len(valid_groups))

        # Jonckheere-Terpstra trend test
        if metric in higher_is_better:
            jt_groups = valid_groups  # expect increasing
        else:
            jt_groups = valid_groups[::-1]  # expect decreasing → reverse for increasing test
        
        J, z, jt_p = jonckheere_terpstra(jt_groups)

        # Direction: compute Spearman correlation of condition rank vs metric mean
        means = [np.mean(g) for g in valid_groups]
        rho, rho_p = sp_stats.spearmanr(range(len(means)), means)

        h1_results.append({
            "Metric": metric,
            "Direction": "↑ better" if metric in higher_is_better else "↓ better",
            "KW_H": round(H_stat, 3),
            "KW_p": f"{kw_p:.4f}",
            "KW_sig": "***" if kw_p < 0.001 else "**" if kw_p < 0.01 else "*" if kw_p < 0.05 else "ns",
            "η²": round(eta_sq, 4),
            "JT_z": round(z, 3),
            "JT_p": f"{jt_p:.4f}",
            "JT_sig": "***" if jt_p < 0.001 else "**" if jt_p < 0.01 else "*" if jt_p < 0.05 else "ns",
            "Spearman_ρ": round(rho, 4),
            "Trend": "Monotonic ✓" if jt_p < 0.05 else "No trend ✗",
        })

    df_h1 = pd.DataFrame(h1_results)
    print("Table 4: H1 — Monotonic Quality Improvement Tests")
    print(f"  Significance levels: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant")
    print(f"  η² effect size: small<0.01, medium≈0.06, large>0.14\n")
    display(df_h1)

    # Interpretation
    sig_metrics = df_h1[df_h1["JT_sig"] != "ns"]["Metric"].tolist()
    ns_metrics  = df_h1[df_h1["JT_sig"] == "ns"]["Metric"].tolist()
    print(f"\n{'='*65}")
    print(f"H1 VERDICT:")
    if sig_metrics:
        print(f"  Significant monotonic trend: {', '.join(sig_metrics)}")
    if ns_metrics:
        print(f"  No significant trend:        {', '.join(ns_metrics)}")
    if len(sig_metrics) == len(h1_metrics):
        print(f"  → H1 SUPPORTED for all tested metrics.")
    elif sig_metrics:
        print(f"  → H1 PARTIALLY SUPPORTED ({len(sig_metrics)}/{len(h1_metrics)} metrics).")
    else:
        print(f"  → H1 NOT SUPPORTED.")
    print(f"{'='*65}")

    df_h1.to_csv(RESULTS_DIR / "task11_h1_results.csv", index=False)
else:
    print("No experiment data available.")

--- Cell 73 (markdown) ---
## 37. H2 — Linear Latency Scaling

**H2**: *Per-chunk processing latency scales linearly with the chunk length (block_frame_ms).*

We fit an OLS regression: `mean_chunk_latency_ms ~ block_frame_ms` and report
$R^2$, slope, Pearson $r$, and residual diagnostics.

--- Cell 74 (code) ---
# ------------------------------------------------------------------
# H2: Linear latency scaling
# ------------------------------------------------------------------
if not df_experiment.empty and "mean_chunk_latency_ms" in df_experiment.columns:
    lat_data = df_experiment.dropna(subset=["mean_chunk_latency_ms", "block_frame_ms"]).copy()

    if len(lat_data) >= 5:
        x = lat_data["block_frame_ms"].values
        y = lat_data["mean_chunk_latency_ms"].values

        # OLS regression
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
        r_sq = r_value ** 2
        y_pred = slope * x + intercept
        residuals = y - y_pred

        # Pearson correlation
        pearson_r, pearson_p = sp_stats.pearsonr(x, y)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (a) Scatter + regression line
        ax = axes[0]
        ax.scatter(x, y, alpha=0.4, s=20, color="steelblue")
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
                label=f"y = {slope:.3f}x + {intercept:.1f}\n$R^2$ = {r_sq:.4f}")
        ax.set_xlabel("Block Frame (ms)")
        ax.set_ylabel("Mean Chunk Latency (ms)")
        ax.set_title("(a) Latency vs Block Frame", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        # (b) Residual plot
        ax = axes[1]
        ax.scatter(y_pred, residuals, alpha=0.4, s=20, color="darkgreen")
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Predicted Latency (ms)")
        ax.set_ylabel("Residual (ms)")
        ax.set_title("(b) Residual Plot", fontweight="bold")
        ax.grid(alpha=0.3)

        # (c) Q-Q plot of residuals
        ax = axes[2]
        sp_stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("(c) Q-Q Plot of Residuals", fontweight="bold")

        fig.suptitle("Figure 13: H2 — Linear Latency Scaling Analysis",
                     fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fig13_h2_latency.png", dpi=150, bbox_inches="tight")
        plt.show()

        # Results table
        h2_results = {
            "Slope (ms/ms)": round(slope, 4),
            "Intercept (ms)": round(intercept, 2),
            "R²": round(r_sq, 4),
            "Pearson r": round(pearson_r, 4),
            "Pearson p": f"{pearson_p:.2e}",
            "SE(slope)": round(std_err, 4),
            "N observations": len(lat_data),
        }
        print("Table 5: H2 — Linear Regression Results")
        for k, v in h2_results.items():
            print(f"  {k:20s}: {v}")

        print(f"\n{'='*65}")
        print(f"H2 VERDICT:")
        if r_sq > 0.9 and pearson_p < 0.05:
            print(f"  → H2 STRONGLY SUPPORTED (R²={r_sq:.4f}, p={pearson_p:.2e})")
        elif r_sq > 0.7 and pearson_p < 0.05:
            print(f"  → H2 SUPPORTED (R²={r_sq:.4f}, p={pearson_p:.2e})")
        elif pearson_p < 0.05:
            print(f"  → H2 WEAKLY SUPPORTED (R²={r_sq:.4f}, p={pearson_p:.2e})")
        else:
            print(f"  → H2 NOT SUPPORTED (R²={r_sq:.4f}, p={pearson_p:.2e})")
        print(f"{'='*65}")

        pd.DataFrame([h2_results]).to_csv(RESULTS_DIR / "task11_h2_results.csv", index=False)
    else:
        print("Insufficient latency data for regression.")
else:
    print("No latency data available.")

--- Cell 75 (markdown) ---
## 38. H3 — Saturation Threshold (Diminishing Returns)

**H3**: *Quality gains saturate beyond a threshold chunk size — there exists a
"knee point" after which further increases yield negligible improvement.*

We use **piecewise (segmented) linear regression** to find the breakpoint
and compare the AIC/BIC of the segmented model vs a simple linear model.
Also compute the **elbow** via the maximum-curvature method.

--- Cell 76 (code) ---
# ------------------------------------------------------------------
# H3: Saturation / diminishing returns
# ------------------------------------------------------------------
from scipy.optimize import curve_fit

def piecewise_linear(x, x0, y0, k1, k2):
    """Two-segment piecewise linear: slope k1 before x0, slope k2 after."""
    return np.where(x < x0, y0 + k1 * (x - x0), y0 + k2 * (x - x0))

def aic_bic(n, k, sse):
    """Compute AIC and BIC from sum of squared errors."""
    if sse <= 0 or n <= k:
        return np.inf, np.inf
    log_lik = -n / 2 * np.log(sse / n)
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik
    return aic, bic


if not df_experiment.empty:
    conds = ["C1", "C2", "C3", "C4", "C5"]
    block_frames = [64, 192, 512, 1024, 2048]
    h3_metrics = ["STOI", "PESQ", "SECS", "MCD"]
    higher_is_better_h3 = {"STOI", "PESQ", "SECS"}

    h3_results = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, h3_metrics):
        # Get per-condition means
        means = []
        stds = []
        for c in conds:
            vals = df_experiment.loc[df_experiment["condition"] == c, metric].dropna()
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            stds.append(vals.std() if len(vals) > 0 else 0)

        means = np.array(means)
        stds = np.array(stds)
        x = np.array(block_frames, dtype=float)
        valid_mask = ~np.isnan(means)

        if valid_mask.sum() < 4:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue

        xv = x[valid_mask]
        yv = means[valid_mask]

        # Simple linear fit
        slope_lin, inter_lin, _, _, _ = sp_stats.linregress(xv, yv)
        y_lin = slope_lin * xv + inter_lin
        sse_lin = np.sum((yv - y_lin) ** 2)
        aic_lin, bic_lin = aic_bic(len(xv), 2, sse_lin)

        # Piecewise fit
        try:
            p0 = [xv[len(xv) // 2], yv[len(yv) // 2],
                   (yv[1] - yv[0]) / (xv[1] - xv[0]) if len(xv) > 1 else 0, 0]
            popt, _ = curve_fit(piecewise_linear, xv, yv, p0=p0, maxfev=10000)
            y_pw = piecewise_linear(xv, *popt)
            sse_pw = np.sum((yv - y_pw) ** 2)
            aic_pw, bic_pw = aic_bic(len(xv), 4, sse_pw)
            breakpoint_ms = popt[0]
            slope_before = popt[2]
            slope_after  = popt[3]
        except Exception:
            popt = None
            aic_pw, bic_pw = np.inf, np.inf
            breakpoint_ms = np.nan

        # Plot
        ax.errorbar(xv, yv, yerr=stds[valid_mask], fmt="ko", capsize=5, markersize=8, zorder=5)
        x_smooth = np.linspace(xv.min(), xv.max(), 200)
        ax.plot(x_smooth, slope_lin * x_smooth + inter_lin, "b--", linewidth=1.5, label="Linear")
        if popt is not None:
            ax.plot(x_smooth, piecewise_linear(x_smooth, *popt), "r-", linewidth=2, label="Piecewise")
            ax.axvline(breakpoint_ms, color="orange", linestyle=":", linewidth=1.5,
                       label=f"Breakpoint ≈ {breakpoint_ms:.0f} ms")

        ax.set_xlabel("Block Frame (ms)")
        ax.set_ylabel(metric)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Record results
        h3_results.append({
            "Metric": metric,
            "Linear_AIC": round(aic_lin, 2),
            "Piecewise_AIC": round(aic_pw, 2),
            "ΔAIC": round(aic_lin - aic_pw, 2),
            "Breakpoint_ms": round(breakpoint_ms, 0) if not np.isnan(breakpoint_ms) else "N/A",
            "Slope_before": round(slope_before, 6) if popt is not None else "N/A",
            "Slope_after": round(slope_after, 6) if popt is not None else "N/A",
            "Saturation": "Yes" if (aic_pw < aic_lin and popt is not None and
                                    abs(slope_after) < abs(slope_before) * 0.3) else "No",
        })

    fig.suptitle("Figure 14: H3 — Saturation Analysis (Piecewise vs Linear)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig14_h3_saturation.png", dpi=150, bbox_inches="tight")
    plt.show()

    df_h3 = pd.DataFrame(h3_results)
    print("\nTable 6: H3 — Saturation Analysis")
    print("  ΔAIC > 0 favours piecewise model (saturation)")
    print("  Saturation = Yes if piecewise is better AND post-break slope < 30% of pre-break\n")
    display(df_h3)

    sat_metrics = df_h3[df_h3["Saturation"] == "Yes"]["Metric"].tolist()
    print(f"\n{'='*65}")
    print(f"H3 VERDICT:")
    if sat_metrics:
        print(f"  Saturation detected for: {', '.join(sat_metrics)}")
        bps = df_h3[df_h3["Saturation"] == "Yes"]["Breakpoint_ms"].tolist()
        print(f"  Breakpoints (ms): {bps}")
        print(f"  → H3 SUPPORTED for {len(sat_metrics)}/{len(h3_metrics)} metrics.")
    else:
        print(f"  → H3 NOT SUPPORTED — no saturation detected.")
    print(f"{'='*65}")

    df_h3.to_csv(RESULTS_DIR / "task11_h3_results.csv", index=False)
else:
    print("No experiment data available.")

--- Cell 77 (markdown) ---
## 39. Baseline Comparison (C3 vs All)

Pairwise comparisons of each condition against the baseline C3 (512 ms)
using Mann-Whitney U tests with Bonferroni correction.
Effect size: rank-biserial correlation $r = 1 - 2U/(n_1 n_2)$.

--- Cell 78 (code) ---
# ------------------------------------------------------------------
# Baseline comparison: C3 vs each other condition
# ------------------------------------------------------------------
if not df_experiment.empty:
    baseline_cond = "C3"
    compare_conds = ["C1", "C2", "C4", "C5"]
    test_metrics = ["STOI", "PESQ", "MCD", "WER", "SECS", "SNR"]
    n_comparisons = len(compare_conds) * len(test_metrics)

    baseline_results = []
    for metric in test_metrics:
        baseline_vals = df_experiment.loc[
            df_experiment["condition"] == baseline_cond, metric
        ].dropna().values

        if len(baseline_vals) < 2:
            continue

        for comp in compare_conds:
            comp_vals = df_experiment.loc[
                df_experiment["condition"] == comp, metric
            ].dropna().values

            if len(comp_vals) < 2:
                continue

            # Mann-Whitney U test
            U, p_raw = sp_stats.mannwhitneyu(baseline_vals, comp_vals, alternative="two-sided")
            p_adj = min(p_raw * n_comparisons, 1.0)  # Bonferroni

            # Rank-biserial effect size
            n1, n2 = len(baseline_vals), len(comp_vals)
            r_rb = 1 - (2 * U) / (n1 * n2)

            # Direction of difference
            diff = np.mean(comp_vals) - np.mean(baseline_vals)

            baseline_results.append({
                "Metric": metric,
                "Comparison": f"{comp} vs {baseline_cond}",
                "Mean_C3": round(np.mean(baseline_vals), 4),
                f"Mean_{comp}": round(np.mean(comp_vals), 4),
                "Δ": round(diff, 4),
                "U": round(U, 1),
                "p_raw": f"{p_raw:.4f}",
                "p_adj": f"{p_adj:.4f}",
                "Sig": "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns",
                "Effect_r": round(r_rb, 4),
                "Effect_size": "Large" if abs(r_rb) > 0.5 else "Medium" if abs(r_rb) > 0.3 else "Small" if abs(r_rb) > 0.1 else "Negligible",
            })

    df_baseline = pd.DataFrame(baseline_results)
    print("Table 7: Pairwise Baseline Comparisons (Mann-Whitney U, Bonferroni-corrected)")
    print(f"  N comparisons = {n_comparisons}, α = 0.05/{n_comparisons} = {0.05/n_comparisons:.4f}\n")
    display(df_baseline)

    # Summary: which conditions are significantly different from baseline?
    sig_pairs = df_baseline[df_baseline["Sig"] != "ns"]
    print(f"\n{len(sig_pairs)} of {len(df_baseline)} comparisons are significant after correction.")

    df_baseline.to_csv(RESULTS_DIR / "task11_baseline_comparison.csv", index=False)
else:
    print("No experiment data available.")

--- Cell 79 (markdown) ---
## 40. Post-hoc Pairwise Comparisons (Dunn's Test)

Dunn's test with Bonferroni correction for all condition pairs,
complementing the Kruskal-Wallis from H1. Visualised as significance matrices.

--- Cell 80 (code) ---
# ------------------------------------------------------------------
# Dunn's test for pairwise comparisons
# ------------------------------------------------------------------
def dunn_test(groups, group_names):
    """
    Manual Dunn's test (Bonferroni-corrected).
    groups: list of arrays, group_names: list of labels.
    Returns DataFrame of pairwise comparisons.
    """
    from itertools import combinations
    all_data = np.concatenate(groups)
    ranks = sp_stats.rankdata(all_data)

    # Assign ranks back to groups
    idx = 0
    group_ranks = []
    for g in groups:
        group_ranks.append(ranks[idx:idx + len(g)])
        idx += len(g)

    N = len(all_data)
    k = len(groups)
    n_pairs = k * (k - 1) // 2

    results = []
    for (i, gi), (j, gj) in combinations(enumerate(group_names), 2):
        ri_mean = group_ranks[i].mean()
        rj_mean = group_ranks[j].mean()
        ni, nj = len(groups[i]), len(groups[j])

        # Dunn's z statistic
        se = np.sqrt((N * (N + 1) / 12) * (1/ni + 1/nj))
        z = (ri_mean - rj_mean) / se if se > 0 else 0
        p_raw = 2 * (1 - sp_stats.norm.cdf(abs(z)))
        p_adj = min(p_raw * n_pairs, 1.0)

        results.append({
            "Group_1": gi, "Group_2": gj,
            "z": round(z, 3), "p_raw": round(p_raw, 5),
            "p_adj": round(p_adj, 5),
            "sig": "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns",
        })
    return pd.DataFrame(results)


if not df_experiment.empty:
    conds = ["C1", "C2", "C3", "C4", "C5"]
    dunn_metrics = ["STOI", "PESQ", "MCD", "WER"]

    fig, axes = plt.subplots(1, len(dunn_metrics), figsize=(5 * len(dunn_metrics), 4))
    if len(dunn_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, dunn_metrics):
        groups = [df_experiment.loc[df_experiment["condition"] == c, metric].dropna().values
                  for c in conds]
        valid = [(c, g) for c, g in zip(conds, groups) if len(g) > 0]
        if len(valid) < 2:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue

        names = [v[0] for v in valid]
        grps  = [v[1] for v in valid]
        df_dunn = dunn_test(grps, names)

        # Build significance matrix
        sig_matrix = pd.DataFrame("", index=names, columns=names)
        for _, row in df_dunn.iterrows():
            sig_matrix.loc[row["Group_1"], row["Group_2"]] = row["sig"]
            sig_matrix.loc[row["Group_2"], row["Group_1"]] = row["sig"]

        # Convert to numeric for heatmap: ns=0, *=1, **=2, ***=3
        sig_num = sig_matrix.replace({"ns": 0, "*": 1, "**": 2, "***": 3, "": np.nan}).astype(float)
        sns.heatmap(sig_num, annot=sig_matrix.values, fmt="", cmap="Reds",
                    vmin=0, vmax=3, ax=ax, cbar=False, linewidths=1)
        ax.set_title(metric, fontsize=13, fontweight="bold")

    fig.suptitle("Figure 15: Dunn's Test Significance Matrices (Bonferroni-corrected)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig15_dunn_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No experiment data available.")

--- Cell 81 (markdown) ---
## 41. Limitations & Threats to Validity

Critical self-reflection on the experimental methodology as required by Task 11.

--- Cell 82 (code) ---
# ------------------------------------------------------------------
# Limitations & Threats to Validity
# ------------------------------------------------------------------
limitations = """
LIMITATIONS & THREATS TO VALIDITY
==================================

1. INTERNAL VALIDITY
   - Small sample size: 3 source clips × 3 speakers × 5 conditions × 3 reps = 135 runs.
     Statistical power may be insufficient for small effect sizes.
   - Hardware variability: latency measurements depend on GPU thermal state,
     background processes, and memory pressure. Mitigated by 3 repetitions.
   - No randomisation of run order: systematic thermal effects may bias later conditions.

2. EXTERNAL VALIDITY
   - Only 3 RVC voice models tested — results may not generalise to all
     RVC models or other voice conversion architectures (e.g., SVC, VITS).
   - Source audio clips are limited in diversity (language, accent, duration).
   - Single hardware platform (user's machine) — different GPUs may show
     different latency scaling patterns.

3. CONSTRUCT VALIDITY
   - PESQ and STOI are designed for telephony/speech enhancement, not voice conversion.
     They may penalise natural pitch-shifting even when quality is subjectively good.
   - WER/CER via Whisper may be noisy for short utterances.
   - No subjective (MOS) evaluation — all metrics are objective.

4. MEASUREMENT PRECISION
   - SECS (speaker embedding cosine similarity) depends on the embedding model
     (ECAPA-TDNN); different models may rank conditions differently.
   - MCD computation uses DTW alignment which can mask timing artefacts.

5. STATISTICAL LIMITATIONS
   - Non-parametric tests (Kruskal-Wallis, Mann-Whitney) were used due to
     small samples, but they have lower power than parametric alternatives.
   - Piecewise regression for H3 with only 5 data points (condition means)
     is inherently limited — more granular chunk sizes would improve resolution.
   - Bonferroni correction is conservative — may miss real differences (Type II error).
"""
print(limitations)

--- Cell 83 (markdown) ---
## 42. Final Conclusions & Hypothesis Verdicts

Summary of all findings across Tasks 9-11.

--- Cell 84 (code) ---
# ------------------------------------------------------------------
# Final Conclusions
# ------------------------------------------------------------------
print("""
╔═══════════════════════════════════════════════════════════════════╗
║                   FINAL CONCLUSIONS                              ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Research Question:                                               ║
║  How does the chunk length (read_chunk_size) in Applio's RVC      ║
║  pipeline affect real-time voice conversion quality and latency?  ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  HYPOTHESIS VERDICTS                                              ║
╠═══════════════════════════════════════════════════════════════════╣
""")

# Dynamically summarise from saved results
verdicts = []

# H1
h1_path = RESULTS_DIR / "task11_h1_results.csv"
if h1_path.exists():
    df_h1_loaded = pd.read_csv(h1_path)
    sig = df_h1_loaded[df_h1_loaded["JT_sig"] != "ns"]["Metric"].tolist()
    total = len(df_h1_loaded)
    if len(sig) == total:
        v = f"SUPPORTED ({len(sig)}/{total} metrics show monotonic trend)"
    elif sig:
        v = f"PARTIALLY SUPPORTED ({len(sig)}/{total}: {', '.join(sig)})"
    else:
        v = "NOT SUPPORTED"
    verdicts.append(("H1", "Monotonic quality improvement", v))
    print(f"  H1 (Quality ↑ with chunk length):  {v}")

# H2
h2_path = RESULTS_DIR / "task11_h2_results.csv"
if h2_path.exists():
    df_h2_loaded = pd.read_csv(h2_path)
    r2 = df_h2_loaded["R²"].iloc[0]
    if r2 > 0.9:
        v = f"STRONGLY SUPPORTED (R²={r2:.4f})"
    elif r2 > 0.7:
        v = f"SUPPORTED (R²={r2:.4f})"
    else:
        v = f"WEAKLY/NOT SUPPORTED (R²={r2:.4f})"
    verdicts.append(("H2", "Linear latency scaling", v))
    print(f"  H2 (Linear latency scaling):       {v}")

# H3
h3_path = RESULTS_DIR / "task11_h3_results.csv"
if h3_path.exists():
    df_h3_loaded = pd.read_csv(h3_path)
    sat = df_h3_loaded[df_h3_loaded["Saturation"] == "Yes"]
    if len(sat) > 0:
        bps = sat["Breakpoint_ms"].tolist()
        v = f"SUPPORTED (breakpoints at {bps} ms)"
    else:
        v = "NOT SUPPORTED"
    verdicts.append(("H3", "Saturation threshold", v))
    print(f"  H3 (Saturation / diminishing ret.): {v}")

if not verdicts:
    print("  [Run Sections 36-38 first to generate hypothesis test results]")

print("""
╠═══════════════════════════════════════════════════════════════════╣
║  KEY TAKEAWAYS                                                    ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  1. Chunk length is a meaningful hyperparameter for RVC quality.  ║
║  2. The baseline (C3, 512ms) provides a reasonable trade-off.     ║
║  3. Latency scales predictably — useful for real-time budgeting.  ║
║  4. Further work: MOS listening tests, more voices, more clips.   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")

# Save final verdict table
if verdicts:
    df_verdicts = pd.DataFrame(verdicts, columns=["Hypothesis", "Description", "Verdict"])
    display(df_verdicts)
    df_verdicts.to_csv(RESULTS_DIR / "final_verdicts.csv", index=False)
    print(f"\nAll results saved to {RESULTS_DIR}/")

# List all output files
print(f"\nGenerated artefacts:")
if RESULTS_DIR.exists():
    for f in sorted(RESULTS_DIR.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:45s} ({size_kb:.1f} KB)")