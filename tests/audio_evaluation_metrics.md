# Audio Evaluation Metrics Reference
## Real-Time Audio-Visual Deepfake Generation: Quality-Latency Tradeoff Analysis

**Prepared by:** Aron Wojciechowicz (Infrastructure & Audio Support)  
**For:** Alex Krasnoshtanov (Audio Lead Researcher)  
**Research Group:** Y2C Real-Time Deepfake Cohort  
**Date:** 2026-02-09  
**Version:** 1.0

---

## Purpose

This document defines the audio evaluation metrics used to assess deepfake generation quality across two experimental conditions: (1) baseline offline generation (Wav2Lip) and (2) real-time generation systems (MuseTalk, LivePortrait, RVC pipeline). Metrics are grouped into three categories based on what dimension of audio quality they capture.

All metrics apply to both conditions unless explicitly noted. The distinction between baseline and real-time evaluation lies not in which metrics are used, but in how results are interpreted: baseline results establish the upper bound of achievable quality, while real-time results quantify the degradation introduced by latency constraints.

---

## Category 1: Speaker Similarity (Voice Identity Preservation)

These metrics measure how well the voice conversion system preserves or matches the target speaker's vocal identity. In our context, this answers: "Does the deepfake sound like the target person?"

### 1.1 Speaker Embedding Cosine Similarity (SECS)

**What it measures:** The angular distance between speaker embedding vectors extracted from the converted audio and the target speaker's reference audio. This captures overall speaker identity similarity — timbre, vocal tract characteristics, and speaking style.

**How it works:** A pre-trained speaker verification model (e.g., ECAPA-TDNN, WavLM-base-SV, or Resemblyzer with GE2E loss) encodes both the converted output and the target reference into fixed-dimensional embedding vectors. Cosine similarity is then computed between these vectors.

**Formula:**

$$\text{SECS} = \frac{\mathbf{e}_{\text{converted}} \cdot \mathbf{e}_{\text{target}}}{\|\mathbf{e}_{\text{converted}}\| \cdot \|\mathbf{e}_{\text{target}}\|}$$

Where $\mathbf{e}$ denotes the speaker embedding vector.

**Range:** -1 to 1 (practically 0 to 1 for meaningful comparisons).

**Interpretation:**
- \> 0.85: Excellent — near-indistinguishable from target speaker
- 0.70–0.85: Good — clearly recognisable as target speaker
- 0.50–0.70: Moderate — some speaker characteristics preserved
- < 0.50: Poor — significant identity loss

**Tool:** `Resemblyzer` (GE2E-based) or `SpeechBrain` (ECAPA-TDNN). We recommend ECAPA-TDNN via SpeechBrain as it is the current standard in voice conversion literature (2024–2025).

**Why it matters for our research:** SECS is the primary metric for evaluating whether real-time voice conversion degrades speaker identity compared to offline processing. If quantisation (Aron's sub-question) or latency constraints (Alex's focus) reduce speaker similarity, SECS will capture it directly.

**Important note on encoder choice:** Results are not comparable across different speaker encoders. The team must standardise on a single encoder model and document the exact version. We recommend `speechbrain/spkrec-ecapa-voxceleb` as the reference encoder for all experiments.

---

### 1.2 Mel Cepstral Distortion (MCD)

**What it measures:** The spectral distance between the converted audio and the target reference audio in the mel-frequency cepstral domain. This captures how closely the spectral envelope (the "shape" of the sound) of the converted speech matches the target.

**How it works:** Mel-frequency cepstral coefficients (MFCCs) are extracted from both the converted and target audio. The Euclidean distance between corresponding MFCC vectors is computed frame-by-frame, then averaged across all frames. Dynamic Time Warping (DTW) alignment is applied first to handle timing differences between the two utterances.

**Formula:**

$$\text{MCD} = \frac{10}{\ln 10} \sqrt{2 \sum_{k=1}^{K} (c_k^{\text{converted}} - c_k^{\text{target}})^2} \quad \text{(dB)}$$

Where $c_k$ denotes the $k$-th mel cepstral coefficient and $K$ is the number of coefficients (typically 13 or 24).

**Range:** 0 to ~15 dB (lower is better).

**Interpretation:**
- < 4.0 dB: Excellent — near-natural quality
- 4.0–6.0 dB: Good — minor spectral differences
- 6.0–8.0 dB: Moderate — noticeable spectral artifacts
- \> 8.0 dB: Poor — significant spectral distortion

**Tool:** `librosa` for MFCC extraction + custom DTW alignment, or `pymcd` package.

**Why it matters for our research:** MCD captures spectral degradation that may not be reflected in speaker embeddings. A system could maintain speaker identity (high SECS) while introducing spectral artifacts that sound unnatural (high MCD). This is especially relevant when comparing FP32 vs. INT8 quantisation, where numerical precision loss may introduce subtle spectral noise.

**Limitation:** MCD requires a paired reference (same sentence spoken by the target). For zero-shot voice conversion scenarios where no parallel data exists, MCD may require DTW alignment, which introduces its own errors. SECS is preferred when parallel data is unavailable.

---

### 1.3 F0 Correlation and F0 RMSE (Pitch Preservation)

**What it measures:** How well the fundamental frequency (pitch) contour of the converted speech matches the target or preserves the source prosody. This captures whether the converted speech sounds natural in terms of intonation, stress patterns, and speaking rhythm.

**How it works:** F0 (fundamental frequency) is extracted from both signals using a pitch tracker (e.g., CREPE, WORLD, or pYIN). Two sub-metrics are computed:

- **F0 Pearson Correlation:** Measures whether pitch rises and falls occur at the same times. High correlation means natural prosody is preserved.
- **F0 RMSE:** Measures the absolute pitch distance. Low RMSE means the actual pitch values are close.

**Range:**
- F0 Correlation: -1 to 1 (higher is better; > 0.8 indicates good preservation)
- F0 RMSE: 0 to ~100 Hz (lower is better; < 20 Hz is good for same-gender conversion)

**Tool:** `librosa.pyin()` or `CREPE` for F0 extraction, `scipy.stats.pearsonr` for correlation.

**Why it matters for our research:** Voice conversion can introduce pitch artifacts — monotone output, pitch jumps, or unnatural vibrato. These are often perceptually jarring even when spectral quality is otherwise acceptable. F0 metrics help isolate pitch-specific degradation from overall quality.

---

## Category 2: Speech Intelligibility and Content Preservation

These metrics measure whether the spoken content is preserved after voice conversion. In our context, this answers: "Can you still understand what is being said?"

### 2.1 Word Error Rate (WER)

**What it measures:** The proportion of words incorrectly transcribed by an automatic speech recognition (ASR) system when processing the converted audio, compared to the known ground-truth transcript. This is a proxy for speech intelligibility — if an ASR system struggles to transcribe it, humans likely will too.

**How it works:** The converted audio is passed through a pre-trained ASR model (we recommend OpenAI Whisper `large-v3`). The resulting transcript is compared to the ground-truth transcript using edit distance.

**Formula:**

$$\text{WER} = \frac{S + D + I}{N} \times 100\%$$

Where $S$ = substitutions, $D$ = deletions, $I$ = insertions, and $N$ = total words in the reference.

**Range:** 0% to >100% (lower is better; can exceed 100% if many insertions occur).

**Interpretation:**
- < 5%: Excellent — near-perfect intelligibility
- 5–15%: Good — minor transcription errors, mostly intelligible
- 15–30%: Moderate — noticeable content degradation
- \> 30%: Poor — significant intelligibility loss

**Tool:** `openai-whisper` (large-v3) for transcription, `jiwer` package for WER computation.

**Why it matters for our research:** WER is our primary intelligibility metric. Real-time systems under latency pressure may produce audio artifacts that degrade intelligibility even when speaker similarity is maintained. INT8 quantisation specifically risks introducing artifacts that affect consonant clarity and phoneme boundaries.

**Important note on ASR model choice:** As with SECS, WER results depend heavily on which ASR model is used. The team must standardise on Whisper `large-v3` for all experiments. Using a different ASR model produces non-comparable results.

---

### 2.2 Character Error Rate (CER)

**What it measures:** Same as WER but computed at the character level rather than word level. This provides finer-grained measurement of intelligibility, particularly useful for short utterances where a single word error produces a misleadingly high WER.

**Formula:**

$$\text{CER} = \frac{S_c + D_c + I_c}{N_c} \times 100\%$$

Where subscript $c$ denotes character-level operations.

**Range:** 0% to >100% (lower is better).

**Interpretation:** Generally follows the same quality bands as WER but with lower absolute values (a single word substitution may affect 5+ characters in WER but fewer proportionally in CER).

**Tool:** `jiwer` package (supports both WER and CER).

**Why it matters for our research:** CER supplements WER for cases where word-level comparison is too coarse. For our dataset (FakeAVCeleb, 10–30 second clips with varied sentence lengths), CER provides a more stable metric when clip durations vary significantly.

---

### 2.3 Short-Time Objective Intelligibility (STOI)

**What it measures:** A frame-level intelligibility estimate that predicts how understandable the converted speech would be to human listeners. Unlike WER (which relies on an ASR model as proxy), STOI directly compares the temporal-spectral envelope of clean and degraded speech signals.

**How it works:** Both signals are decomposed into short-time overlapping segments. For each segment, spectral envelopes are compared using normalised correlation. The final STOI score is the average correlation across all segments and frequency bands.

**Range:** 0 to 1 (higher is better).

**Interpretation:**
- \> 0.90: Excellent intelligibility
- 0.75–0.90: Good intelligibility
- 0.60–0.75: Fair — some effort required to understand
- < 0.60: Poor — largely unintelligible

**Tool:** `pystoi` package.

**Requirement:** STOI requires a paired reference signal (the original clean audio before conversion). Both signals must be time-aligned and have the same sampling rate.

**Why it matters for our research:** STOI captures intelligibility degradation that WER might miss (or that WER captures too coarsely). It is particularly sensitive to the types of distortions introduced by low-precision inference (noise floor, quantisation artifacts) and real-time buffering (audio dropouts, chunk boundary artifacts).

---

## Category 3: Audio-Visual Synchronisation

These metrics measure the temporal alignment between the audio output and the corresponding visual lip movements in the generated video. In our context, this answers: "Do the lips match the speech?"

**Note:** These metrics require both audio and video outputs. They evaluate the combined A/V pipeline, not audio alone. Alex's audio pipeline feeds into the visual pipeline; synchronisation errors can originate from either side.

### 3.1 Lip-Sync Error Distance (LSE-D) — PRIMARY METRIC

**What it measures:** The temporal offset (in milliseconds) between the audio signal and the corresponding lip movements in the generated video. This is the most direct measure of audio-visual synchronisation quality.

**How it works:** SyncNet (Chung & Zisserman, 2017) extracts audio features (mel-spectrograms) and visual features (lip region crops) in short windows. It computes the distance between audio and visual feature vectors across a range of temporal offsets. The offset with minimum distance is the estimated lip-sync error.

**Range:** 0 to ~500 ms (lower is better).

**Interpretation:**
- < 40 ms: Excellent — imperceptible to humans
- 40–80 ms: Acceptable — detectable under scrutiny but natural enough
- 80–120 ms: Noticeable — most viewers detect something "off"
- \> 120 ms: Distracting — clearly out of sync

**Tool:** Pre-trained SyncNet model (available via the `syncnet_python` repository).

**Why this is our primary A/V metric:** LSE-D directly quantifies the synchronisation quality that is central to the group's research question. The 80 ms threshold identified in Sub-Q4 (perceptual study) maps directly to this metric's scale. All individual sub-questions should report LSE-D to enable cross-comparison.

---

### 3.2 Lip-Sync Error Confidence (LSE-C)

**What it measures:** The confidence score from SyncNet indicating how well audio and visual streams are aligned. While LSE-D measures the magnitude of misalignment, LSE-C indicates how certain the model is about the alignment quality.

**How it works:** Computed from the same SyncNet forward pass as LSE-D. The confidence score reflects the similarity between the best-matching audio and visual feature vectors.

**Range:** 0 to ~10 (higher is better; exact scale depends on SyncNet version).

**Interpretation:**
- \> 7.0: Strong synchronisation confidence
- 4.0–7.0: Moderate confidence
- < 4.0: Low confidence — likely desynchronised

**Tool:** Same SyncNet model as LSE-D (computed jointly).

**Why it matters for our research:** LSE-C supplements LSE-D by providing a confidence dimension. A low LSE-D (small offset) combined with low LSE-C (low confidence) may indicate that synchronisation appears acceptable by chance rather than genuine alignment. Both should be reported together.

---

### 3.3 Audio-Visual Offset (AV-Offset)

**What it measures:** The raw temporal offset (in frames or milliseconds) between the audio stream and the video stream in the final output. Unlike LSE-D which uses learned features, AV-Offset is a signal-level measurement of when audio events (phoneme onsets) occur relative to visual events (lip movement onsets).

**How it works:** Audio onset detection (using energy or spectral flux) is compared against visual onset detection (using optical flow or lip landmark velocity) across the full video. The systematic offset between the two streams is computed.

**Range:** Negative values (audio leads video) to positive values (video leads audio), in milliseconds.

**Interpretation:**
- -20 to +40 ms: Imperceptible to humans (note: humans are more tolerant of audio leading video than the reverse)
- ±40–80 ms: Detectable but tolerable
- Beyond ±80 ms: Distracting

**Tool:** Custom implementation using `librosa` (audio onset detection) and `dlib` or `MediaPipe` (lip landmark tracking).

**Why it matters for our research:** AV-Offset isolates systematic pipeline-level delays from content-specific synchronisation issues. Real-time systems often introduce a constant offset due to buffering and processing pipelines. Identifying and reporting this constant offset separately from content-dependent lip-sync errors provides clearer diagnostic information for system optimisation.

---

## Metrics Summary Table

| # | Metric | Category | Range | Direction | Primary Tool | Paired Reference Required? |
|---|--------|----------|-------|-----------|--------------|---------------------------|
| 1 | SECS (Cosine Similarity) | Speaker Similarity | -1 to 1 | Higher = better | SpeechBrain (ECAPA-TDNN) | Yes (target speaker reference) |
| 2 | MCD | Speaker Similarity | 0–15 dB | Lower = better | librosa + DTW | Yes (parallel utterance) |
| 3 | F0 Correlation | Speaker Similarity | -1 to 1 | Higher = better | librosa.pyin / CREPE | Yes (source or target) |
| 4 | F0 RMSE | Speaker Similarity | 0–100 Hz | Lower = better | librosa.pyin / CREPE | Yes (source or target) |
| 5 | WER | Intelligibility | 0–100%+ | Lower = better | Whisper large-v3 + jiwer | Yes (ground truth transcript) |
| 6 | CER | Intelligibility | 0–100%+ | Lower = better | Whisper large-v3 + jiwer | Yes (ground truth transcript) |
| 7 | STOI | Intelligibility | 0–1 | Higher = better | pystoi | Yes (clean source audio) |
| 8 | LSE-D | A/V Synchronisation | 0–500 ms | Lower = better | SyncNet | No (self-contained) |
| 9 | LSE-C | A/V Synchronisation | 0–10 | Higher = better | SyncNet | No (self-contained) |
| 10 | AV-Offset | A/V Synchronisation | ±500 ms | Closer to 0 = better | Custom (librosa + MediaPipe) | No (self-contained) |

---

## Recommended Metric Priority

For the scope of this project, not all metrics carry equal weight. The following prioritisation reflects what is most critical to answering our research questions and what is feasible within the 8-week block.

**Tier 1 — Must implement (core metrics):**
- **SECS** — Primary speaker similarity metric. Essential for evaluating voice conversion quality across all conditions.
- **WER** — Primary intelligibility metric. Essential for confirming that conversion does not destroy speech content.
- **LSE-D + LSE-C** — Primary A/V sync metrics. Essential for the group's central research question on quality-latency tradeoff.

**Tier 2 — Should implement (supporting metrics):**
- **MCD** — Adds spectral-level detail to complement SECS. Particularly relevant for quantisation experiments (FP32 vs INT8).
- **F0 Correlation** — Captures pitch preservation, a known weakness of real-time voice conversion.
- **CER** — Supplements WER for short utterances.

**Tier 3 — Nice to have (if time permits):**
- **STOI** — Strong metric but requires careful signal alignment. Implement only if paired audio alignment is already in place.
- **F0 RMSE** — Supplements F0 Correlation with absolute pitch error.
- **AV-Offset** — Supplements SyncNet metrics with signal-level offset measurement.

---

## Standardisation Requirements

To ensure results are comparable across team members' experiments, the following must be fixed for all measurements:

1. **Speaker encoder:** `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN) for all SECS measurements.
2. **ASR model:** OpenAI Whisper `large-v3` for all WER/CER measurements.
3. **SyncNet version:** Document the exact checkpoint used; all team members must use the same one.
4. **Sampling rate:** All audio resampled to 16 kHz before metric computation (standard for speech processing).
5. **Audio format:** WAV (PCM 16-bit) for metric computation; no lossy compression in the evaluation path.
6. **MFCC configuration (for MCD):** 24 coefficients, 25 ms window, 10 ms hop, mel filterbank with 80 channels.
7. **Random seeds:** Fixed for any stochastic components in the pipeline.

---

## Baseline vs. Real-Time: How Metrics Are Applied

### Baseline Evaluation (Wav2Lip, offline)

The baseline represents the "best achievable quality" with no time constraints. All metrics are computed on Wav2Lip outputs processed offline. These values establish the upper bound against which real-time systems are compared.

**What to report:**
- All Tier 1 and Tier 2 metrics on the full test set (100 videos from FakeAVCeleb subset).
- Mean ± standard deviation for each metric.
- Per-video scores saved to CSV for statistical comparison.

### Real-Time Evaluation (MuseTalk, LivePortrait, RVC pipeline)

Real-time outputs are generated under latency constraints (<200 ms target). The same metrics are computed, and results are compared to the baseline using paired statistical tests (paired t-test or Wilcoxon signed-rank test) since both conditions process the same input videos.

**What to report:**
- All Tier 1 and Tier 2 metrics on the same test set.
- Mean ± standard deviation for each metric.
- Per-video scores saved to CSV for statistical comparison.
- **Additionally:** processing latency per video (ms), FPS achieved, and GPU memory usage — to enable quality-latency tradeoff analysis.

### Statistical Comparison

For each metric, the comparison between baseline and real-time conditions should include:
- Paired t-test (or Wilcoxon if normality assumption is violated).
- Effect size (Cohen's d).
- 95% confidence interval for the mean difference.
- Significance level: α = 0.05.

---

## References

Chung, J. S., & Zisserman, A. (2017). Out of Time: Automated Lip Sync in the Wild. *ACCV Workshops*.

Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification. *Interspeech*.

Kubichek, R. F. (1993). Mel-Cepstral Distance Measure for Objective Speech Quality Assessment. *IEEE Pacific Rim Conference on Communications, Computers and Signal Processing*.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML*.

Rix, A. W., Beerends, J. G., Hollier, M. P., & Hekstra, A. P. (2001). Perceptual Evaluation of Speech Quality (PESQ). *ITU-T Recommendation P.862*.

Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2010). A Short-Time Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech. *ICASSP*.

Wan, L., Wang, Q., Papir, A., & Moreno, I. L. (2018). Generalized End-to-End Loss for Speaker Verification. *ICASSP*.
