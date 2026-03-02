# Real-Time Audio-Visual Deepfake Generation: Quality-Latency Tradeoff Analysis

## Comprehensive Research Plan — Draft v2.0

**Research Group:** Y2C 2025-26 (Group O1)  
**Institution:** Breda University of Applied Sciences (BUas), ADS&AI Year 2, Block C  
**Generated:** February 25, 2026 | Week 3  
**Main Deadline:** Proposal Friday Feb 27 17:00 | Final Submission April 3, 2026  
**GitHub:** https://github.com/OleksiiKrasnoshtanov240247/Y2C-2025-2026  
**Trello:** https://trello.com/b/gq91GfrN/y2c-2025-26-team  

---

## TEAM COMPOSITION

| Member | Role | Primary Focus | Sprint Role |
|--------|------|---------------|-------------|
| **Alex Krasnoshtanov** | Audio Lead Researcher | RVC chunk-length tradeoff | Journal Club Speaker (Sprint 2) |
| **Filipp Lotsmanov** | Video Lead Researcher | Face-swap visual fidelity vs. latency | Journal Club Speaker (Sprint 1) |
| **Danil Sysenko** | QA & User Study Researcher | MOS perceptual study, system validation | Scrum Master Sprint 1 |
| **Maksym Steshkin** | Video Support & Metrics Researcher | Objective evaluation metrics, statistical benchmarking | — |
| **Aron Wojciechowicz** | Infrastructure & Audio Support | PTQ optimization, environment, data management | Scrum Master Sprint 2 |

---

## 1. WHAT ARE WE RESEARCHING?

### 1.1 The Core Problem

Real-time deepfake generation systems must simultaneously produce convincing audio-visual output while operating within strict latency constraints. This creates a fundamental engineering tension: every millisecond spent improving output quality is a millisecond added to the processing pipeline. In offline scenarios (pre-recorded content), systems can spend minutes refining a single second of video. In real-time scenarios (live video calls, streaming), the entire pipeline from audio input to video output must complete in under 200 milliseconds to maintain the illusion of natural conversation.

This tradeoff is not well-quantified in existing literature. Most deepfake research evaluates either offline quality (where latency is irrelevant) or real-time capability (where quality is reported but not systematically compared against offline baselines). No study has systematically mapped the space between these two extremes to answer the question: how much quality do you sacrifice for real-time performance, and where exactly does that sacrifice become perceptible to humans?

### 1.2 Main Research Question

> **"How does the quality-latency tradeoff in real-time deepfake generation systems affect audio-visual synchronization quality, and what are the technical and perceptual boundaries of acceptable real-time performance?"**

### 1.3 What We Specifically Investigate

The research examines three interrelated dimensions of the quality-latency tradeoff.

**Audio-Visual Synchronization Quality** is the primary quality dimension. This encompasses lip-sync accuracy (whether visible mouth shapes match the produced phonemes), temporal alignment between audio and visual streams measured in milliseconds of offset, and phoneme-to-viseme correspondence — the degree to which the generated face produces the correct mouth shape at the correct time. Synchronization quality is the most perceptually salient indicator of deepfake quality; humans detect audio-visual misalignment faster than they detect static visual artifacts.

**Visual Quality** covers facial feature preservation (does the generated face look like the target identity?), texture realism (are there visible artifacts, blurring, or unnatural skin texture?), temporal consistency (does the face remain stable frame-to-frame or does it jitter and warp?), and artifact presence (ghosting, boundary artifacts around face edges, teeth rendering failures). Visual quality degrades as latency budgets tighten because systems must use smaller models, lower resolutions, or fewer refinement steps.

**System Performance** captures the engineering side: end-to-end processing latency in milliseconds, throughput in frames per second, computational efficiency (FLOPs per frame), GPU memory usage, and model size. These metrics define the constraints within which quality must be achieved.

### 1.4 The Two-Component Pipeline

Real-time audio-visual deepfakes operate as a two-stage pipeline, and understanding this architecture is essential to understanding our research design.

**Stage 1 — Audio Processing.** Raw audio (microphone input or audio file) is processed by an audio encoder (typically Whisper, Wav2Vec2, or HuBERT/ContentVec) which extracts semantic and acoustic features. For voice conversion specifically, the audio is transformed by a voice conversion model (RVC, Seed-VC, or similar) that maps the source speaker's voice characteristics to a target speaker's voice. The output is either audio embeddings (for driving lip-sync generation) or converted audio waveform (for playback).

**Stage 2 — Visual Generation.** Audio embeddings drive a visual generation model that produces lip-synced video frames. Different architectures approach this differently: Wav2Lip directly generates mouth regions from audio features; MuseTalk operates in a latent space, inpainting the lower face region; LivePortrait uses implicit keypoints to animate a reference face. Each architecture makes different quality-latency tradeoffs.

The full pipeline: `Audio Input → Audio Encoder → Feature Extraction → Voice Conversion (optional) → Audio Embeddings → Visual Generator → Face Rendering → Video Output`

Each component in this pipeline contributes to total latency and may independently degrade quality. Our five sub-questions collectively characterize how different interventions at different pipeline stages affect the overall tradeoff.

### 1.5 Research Boundaries

**In scope:** Talking-head style videos (face and audio), audio-visual synchronization as the primary quality dimension, pre-trained models used in inference mode (no training from scratch), controlled dataset comparisons using FakeAVCeleb, quantitative objective metrics supplemented by subjective perceptual evaluation, and post-training optimization techniques.

**Out of scope:** Full-body deepfakes, training new model architectures, real-time voice cloning from scratch (we use pre-trained voice conversion), detection methods (this is a generation study, not a detection study), and video-to-video face-swap without audio (we focus specifically on audio-visual synchronization).

---

## 2. INDIVIDUAL SUB-QUESTIONS (5 DISTINCT RESEARCH TRACKS)

Each sub-question represents a complete, independent research study that contributes a unique analytical perspective to the group's main research question. Every researcher produces their own proposal, research notebook (4 sections per ILO 7.3A–D), and presentation contribution.

---

### 🔬 SUB-QUESTION 1: Audio Chunk-Length Impact on Voice Conversion Quality

**Researcher:** Alex Krasnoshtanov (Audio Lead)

#### Research Question

> "How does the input buffer size (chunk length) impact voice similarity (CosSim) and processing latency in RVC-based real-time voice conversion?"

#### What Is Being Investigated

The fundamental tradeoff in real-time audio processing is the buffer (chunk) size. When voice conversion operates in streaming mode, audio is not processed all at once but in discrete chunks. A larger chunk gives the model more context — more audio samples to analyze — resulting in higher-quality conversion. But a larger chunk also means more samples must accumulate before processing can begin, which increases latency. A smaller chunk enables lower latency but provides less context, potentially degrading voice similarity.

The independent variable is chunk length, measured in samples or milliseconds, tested across five conditions (e.g., 128, 256, 512, 1024, 2048 samples at 16kHz, corresponding to approximately 8ms, 16ms, 32ms, 64ms, and 128ms). The dependent variables are cosine similarity (CosSim) between the converted voice and the target speaker embedding (measuring how much the converted voice sounds like the target), and processing latency per chunk in milliseconds.

#### Why This Matters

Voice conversion is the audio counterpart to face-swap in the visual domain, and it is comparatively understudied. Most real-time deepfake research focuses on video generation latency while treating audio as a solved problem. However, the audio pipeline introduces its own quality-latency constraints that directly affect the overall system. If the voice conversion introduces audible artifacts or loses speaker similarity at low latencies, the entire deepfake becomes less convincing — even if the visual component is perfect. Alex's research quantifies exactly where this audio quality boundary lies.

Furthermore, the chunk-length tradeoff has practical implications for system design: knowing the minimum viable chunk length tells engineers the theoretical lower bound on audio latency for acceptable quality, which constrains the entire pipeline's real-time budget.

#### Hypotheses

**H1 (Monotonic Quality Improvement):** Larger chunk lengths yield statistically significantly higher cosine similarity scores, following a monotonic relationship — more context produces better voice conversion quality.

**H2 (Linear Latency):** Processing latency increases linearly with chunk length, meaning each additional millisecond of input buffer adds a proportional amount of processing time.

**H3 (Saturation Threshold):** Cosine similarity improvements saturate beyond a critical chunk length — there exists a threshold beyond which adding more context provides diminishing returns in voice similarity, making further increases in chunk size wasteful from a latency perspective.

#### Methodology

**System:** RVC (Retrieval-based Voice Conversion) via the Applio fork, using a pre-trained DonaldTrump.pth model with corresponding index file. The trained model serves as the high-quality offline baseline (full-file processing, no chunking).

**Dataset:** 100 videos from the FakeAVCeleb subset, with audio extracted at 16kHz WAV. Each audio clip is 10–30 seconds of clear speech.

**Experimental Protocol:**
1. Process each audio file through RVC at full length (baseline condition, no chunking).
2. Process each audio file through RVC at five chunk-length conditions: C1 (128 samples / ~8ms), C2 (256 / ~16ms), C3 (512 / ~32ms), C4 (1024 / ~64ms), C5 (2048 / ~128ms).
3. For each output, compute CosSim using ECAPA-TDNN speaker embeddings and record processing latency per chunk.
4. Compare all chunk conditions against the full-length baseline.

**Measurements:**

| Metric | Tool | Purpose |
|--------|------|---------|
| Cosine Similarity (CosSim) | ECAPA-TDNN (SpeechBrain) | Voice similarity to target speaker |
| Processing Latency (ms/chunk) | Python time.perf_counter() | Audio processing speed |
| Word Error Rate (WER) | Whisper ASR | Speech intelligibility preservation |
| Signal-to-Noise Ratio (SNR) | scipy | Audio quality degradation |

**Statistical Analysis:**
- One-way repeated-measures ANOVA across the five chunk conditions for CosSim and latency.
- Tukey HSD post-hoc tests for pairwise chunk-length comparisons.
- Pearson correlation between chunk length and CosSim, and between chunk length and latency.
- Segmented (piecewise) regression to identify the saturation threshold where CosSim improvement flattens.
- Effect sizes reported as Cohen's d and η² (eta-squared).

#### Deliverables

- `alex_krasnoshtanov_proposal.pdf` — Individual research proposal with literature review, baseline definition, methodology, and hypotheses.
- `alex_krasnoshtanov_eda.ipynb` — Exploratory data analysis of the audio dataset and pilot RVC outputs.
- `alex_krasnoshtanov_research_notebook.ipynb` — Four-section notebook (Implementation, Results, Analysis, Synthesis).
- Visualization: CosSim-vs-chunk-length curves with confidence intervals, latency-vs-chunk scatter with linear fit, saturation threshold identification plot.

#### Unique Contribution

Alex's work is the only audio-focused analysis in the group. While the other four sub-questions examine visual generation, optimization, metrics, and perception, Alex quantifies the often-overlooked audio component of the quality-latency tradeoff. His saturation threshold finding directly informs the group's main conclusion about where the overall system's "acceptable quality boundary" lies for the audio dimension.

---

### 🔬 SUB-QUESTION 2: Visual Fidelity vs. Real-Time Constraints in Face-Swap Pipelines

**Researcher:** Filipp Lotsmanov (Video Lead)

#### Research Question

> "How does the choice of face-swap architecture (Wav2Lip baseline vs. MuseTalk vs. LivePortrait) affect visual fidelity and lip-sync accuracy under real-time latency constraints?"

#### What Is Being Investigated

Different face-swap architectures make fundamentally different engineering decisions about how to generate video. Wav2Lip directly predicts mouth pixels from audio features — simple but limited in quality. MuseTalk operates in a learned latent space, inpainting the lower face region from audio-conditioned latents — more complex but potentially higher quality at comparable latency. LivePortrait uses implicit keypoints to animate a reference image based on audio-driven motion fields — the most geometrically flexible approach but with its own quality characteristics.

These are not minor implementation differences; they represent fundamentally different approaches to the generation problem, and each trades off visual fidelity, temporal stability, identity preservation, and processing speed in distinct ways. Filipp's research systematically compares these three architectures under identical conditions to determine which design philosophy produces the best quality-latency tradeoff for real-time talking-head deepfakes.

The independent variable is the generation architecture (three levels: Wav2Lip, MuseTalk, LivePortrait). The dependent variables are visual quality metrics (SSIM, PSNR, LPIPS, CSIM), lip-sync accuracy (SyncNet confidence, Lip Error Distance), per-frame processing latency, and FPS throughput.

#### Why This Matters

Architecture selection is the single most consequential design decision in building a real-time deepfake system. Optimization techniques (Sub-Q5/Aron) and configuration tuning (Sub-Q1/Alex) operate within the constraints set by the chosen architecture. By establishing which architecture provides the best baseline quality-latency tradeoff, Filipp's research sets the ceiling for what the other sub-questions can achieve through optimization and tuning.

Additionally, no existing study provides a controlled comparison of these three specific architectures under identical hardware, dataset, and evaluation conditions. Published results for each system use different datasets, different hardware, and different metrics, making direct comparison impossible. This research fills that gap.

#### Hypotheses

**H2.1 (Latent-Space Advantage):** MuseTalk, which operates in a latent space, achieves statistically significantly higher visual quality (SSIM, LPIPS) than Wav2Lip at equivalent or lower latency, because latent-space operations are computationally cheaper per quality unit than pixel-space operations.

**H2.2 (Lip-Sync Baseline Strength):** Wav2Lip achieves the highest lip-sync accuracy (SyncNet confidence) despite lower overall visual quality, because it was specifically designed and trained for lip-sync rather than general face animation.

**H2.3 (Temporal Stability Tradeoff):** LivePortrait exhibits superior temporal consistency (lower frame-to-frame jitter) due to its keypoint-based animation approach, but at the cost of higher per-frame latency compared to MuseTalk.

#### Methodology

**Systems:**
- **Wav2Lip** (Prajwal et al., 2020) — Offline baseline, audio-conditioned lip generation.
- **MuseTalk v1.5** (2025) — Real-time latent-space inpainting, MIT license, 30+ FPS reported.
- **LivePortrait** (2024) — Implicit keypoint animation, 12.8ms/frame reported.

**Dataset:** Same 100 FakeAVCeleb videos used across all sub-questions.

**Experimental Protocol:**
1. Process all 100 videos through each system under default optimal settings.
2. Profile each system at the component level: face detection, feature extraction, generation, post-processing.
3. Collect per-frame quality metrics and latency measurements.
4. Generate quality-latency Pareto frontiers for each system.

**Measurements:**

| Metric | Tool | What It Captures |
|--------|------|-----------------|
| SSIM | scikit-image | Structural similarity to ground truth |
| PSNR (dB) | scikit-image | Pixel-level reconstruction quality |
| LPIPS | torch-lpips | Perceptual similarity (learned) |
| CSIM | ArcFace | Identity preservation (face recognition) |
| SyncNet Confidence | Pre-trained SyncNet | Lip-sync accuracy |
| Lip Error Distance (LED) | Audio-visual temporal analysis | Sync offset in milliseconds |
| Per-frame latency (ms) | Python profiling | Processing speed |
| FPS | Frame count / wall time | Throughput |
| GPU VRAM (GB) | nvidia-smi | Memory footprint |

**Statistical Analysis:**
- One-way ANOVA (3 architectures) for each quality metric and latency.
- Post-hoc Bonferroni-corrected pairwise comparisons.
- Two-way analysis: architecture × video difficulty (easy/medium/hard based on motion complexity).
- Pareto frontier construction: plot quality vs. latency for all systems, identify the Pareto-optimal architecture.
- Component contribution analysis: what percentage of total latency comes from each pipeline stage?

#### Deliverables

- Individual research proposal, EDA notebook, and four-section research notebook.
- Architecture comparison tables with statistical significance indicators.
- Component-level latency breakdown visualizations (stacked bar charts per architecture).
- Quality-latency Pareto frontier plot — the core visualization showing which system dominates.

#### Unique Contribution

Filipp provides the only controlled cross-architecture comparison. His component-level profiling reveals which pipeline stages are bottlenecks, directly informing Aron's optimization work (if the visual generator dominates latency, that is where quantization matters most). His Pareto frontier becomes the group's central framework for discussing the quality-latency tradeoff.

---

### 🔬 SUB-QUESTION 3: Perceptual Quality Thresholds in Real-Time Deepfake Generation

**Researcher:** Danil Sysenko (QA & User Study)

#### Research Question

> "At what latency threshold does audio-visual synchronization quality become perceptually unacceptable to human observers, and how do objective metrics correlate with subjective perception?"

#### What Is Being Investigated

Technical metrics (SSIM, PSNR, SyncNet confidence) quantify quality mathematically, but they do not necessarily correspond to what humans actually perceive. A system might score well on SSIM but produce temporally unstable output that humans find jarring. Conversely, certain artifacts that reduce PSNR might be imperceptible to viewers. Danil's research bridges this gap by collecting human judgments and correlating them with objective measurements to answer two critical questions: (1) at what point does quality become unacceptable to humans? and (2) which technical metrics best predict human perception?

The independent variable is quality level, operationalized as outputs generated at different latency targets (producing a gradient from high-quality/high-latency to low-quality/low-latency). The dependent variables are Mean Opinion Score (MOS, 1–5 scale), acceptability rate (percentage of participants rating ≥3), detection rate (percentage correctly identifying content as AI-generated), and the correlation between each objective metric and MOS.

#### Why This Matters

Every technical metric is ultimately a proxy for human perception. Without subjective validation, the group's findings about quality-latency tradeoffs remain purely technical — useful for engineers but insufficient for security researchers, policymakers, or system designers who need to know what humans actually experience. Danil's perceptual threshold finding provides the "ground truth" against which all other sub-questions' metrics are calibrated.

The detection rate component is particularly important for security applications: if real-time deepfakes at a certain quality level are reliably detected by untrained observers, that sets an empirical boundary on the threat they pose.

#### Hypotheses

**H3.1 (Sharp Perceptual Threshold):** Human perception of lip-sync quality degrades sharply (not gradually) below a critical latency threshold, predicted to be 100–150ms. Above this threshold, MOS ratings cluster around 3.5–4.0; below it, MOS drops below 2.5.

**H3.2 (Temporal Over Static):** Temporal consistency metrics (optical flow coherence, frame-to-frame LPIPS) correlate more strongly with human MOS ratings (r > 0.7) than static quality metrics (PSNR, single-frame SSIM), because humans are more sensitive to motion artifacts than to per-frame quality reductions.

**H3.3 (Sync Detection Threshold):** Audio-visual synchronization errors exceeding 80ms (approximately 2–3 frames at 30 FPS) are reliably detected by human observers (detection rate > 75%), establishing an empirical "acceptability boundary" for real-time systems.

**H3.4 (Metric Prediction):** A logistic regression model using the top three objective metrics can predict human acceptability (MOS ≥ 3) with AUC > 0.80, enabling automated quality assessment without future human studies.

#### Methodology

**Stimulus Creation:**
1. Select 20 representative video clips from the FakeAVCeleb test set.
2. Generate each clip at 6 quality levels by varying system configuration (latency targets from 50ms to 500ms+).
3. Total stimuli: 20 clips × 6 conditions = 120 video samples.
4. Include 20 real (unmanipulated) clips as controls.

**Participant Recruitment:**
- 20–30 participants (BUas students and faculty).
- Inclusion: normal or corrected-to-normal vision and hearing.
- Exclusion: participants involved in the research group.
- Compensation: voluntary participation (academic context).

**Evaluation Protocol (MOS Study):**
- Randomized presentation order (Latin square design to control for order effects).
- Each participant rates all stimuli on three scales (1–5): overall quality, lip-sync accuracy, naturalness.
- Binary acceptability question: "Would you believe this is a real video in a video call?" (Yes/No).
- Duration: 20–30 minutes per participant.
- Setting: controlled environment (same screen, same audio setup, same viewing distance).

**Ethics Requirements (Critical Path):**
- BUas Ethics Application must be submitted and approved before data collection.
- Informed consent: participants are told they will view AI-generated content before participating.
- Anonymous: no names, emails, or identifying information linked to responses.
- Right to withdraw at any time without consequence.
- Survey data stored as CSV on university server, no personal data collected.

**Measurements:**

| Metric | Scale | Purpose |
|--------|-------|---------|
| MOS (Mean Opinion Score) | 1–5 continuous | Overall perceived quality |
| Acceptability Rate | 0–100% binary | % rating content as acceptable |
| Detection Rate | 0–100% binary | % correctly identifying as fake |
| Inter-rater Reliability (Cronbach's α) | 0–1 | Agreement between participants |

**Statistical Analysis:**
- Pearson and Spearman correlation: each objective metric vs. MOS.
- Multiple regression: predict MOS from top objective metrics.
- Logistic regression: predict acceptability (binary) from objective metrics.
- ROC analysis: identify which metric best discriminates acceptable from unacceptable quality.
- Threshold detection: segmented regression on MOS-vs-latency curve to identify breakpoint.
- Inter-rater reliability: Cronbach's α to validate measurement consistency.

#### Deliverables

- Individual proposal, EDA notebook, and four-section research notebook.
- Ethics Application documentation (ILO 3.5B compliance).
- MOS study protocol and consent form.
- Correlation matrix: all objective metrics vs. MOS.
- Acceptability threshold identification plot with confidence intervals.
- Metric ranking table: which metrics best predict human perception.

#### Unique Contribution

Danil is the only researcher incorporating human evaluation. His work transforms the group's technical findings into actionable knowledge about perceptual boundaries. The correlation analysis between objective metrics and MOS has direct practical value: if a specific metric (e.g., SyncNet confidence > 0.6) reliably predicts human acceptability, future researchers can skip expensive human studies and use that metric as a proxy.

---

### 🔬 SUB-QUESTION 4: Objective Quality Metrics Comparison for Audio-Visual Deepfake Evaluation

**Researcher:** Maksym Steshkin (Video Support & Metrics)

#### Research Question

> "Which combination of objective evaluation metrics most comprehensively captures the quality degradation in real-time deepfake generation compared to offline baselines, and how do individual metrics respond to different types of quality loss?"

#### What Is Being Investigated

The deepfake evaluation literature uses dozens of metrics, but no consensus exists on which metrics are essential, redundant, or complementary. Some metrics measure overlapping quality dimensions (SSIM and PSNR are highly correlated), while others capture orthogonal aspects (FID measures distributional quality, CSIM measures identity preservation). Researchers often report a grab-bag of metrics without justifying their selection or analyzing inter-metric relationships.

Maksym's research systematically evaluates the existing metric landscape by computing a comprehensive set of metrics across multiple quality conditions and analyzing their correlations, sensitivities, and complementarity. The goal is to identify the minimum sufficient metric set that captures all meaningful quality dimensions without redundancy.

The independent variable is the quality condition — outputs from the three architectures (Filipp's data), at different optimization levels (Aron's data), compared against offline baselines. The dependent variables are the full suite of objective metrics, analyzed for their inter-correlations and sensitivity to different degradation types.

#### Why This Matters

Without a principled metric selection, the group risks either over-measuring (reporting redundant metrics that add noise without insight) or under-measuring (missing quality dimensions that matter). Maksym's analysis provides the measurement foundation for the entire group: his recommended metric set becomes the standard evaluation protocol used by all researchers. Additionally, his sensitivity analysis reveals which metrics detect which types of degradation, enabling targeted quality assessment (e.g., "use metric X when checking temporal stability, metric Y when checking identity preservation").

This work also connects directly to Danil's perceptual study: by comparing metric sensitivity patterns with human perception patterns, the group can identify which objective metrics have genuine perceptual grounding.

#### Hypotheses

**H4.1 (Metric Redundancy):** At least 40% of commonly reported deepfake evaluation metrics are statistically redundant (inter-correlation r > 0.85), and a subset of 4–5 metrics can capture >95% of the variance in a principal component analysis of the full metric space.

**H4.2 (Degradation Specificity):** Different metrics show differential sensitivity to different degradation types — temporal metrics (optical flow consistency) are most sensitive to FPS reduction, pixel-level metrics (PSNR) are most sensitive to compression/quantization artifacts, and perceptual metrics (LPIPS) are most sensitive to identity drift.

**H4.3 (Sync Dominance):** Audio-visual synchronization metrics (SyncNet confidence, LED) contribute the most unique variance (highest unique loading in PCA) compared to visual-only metrics, because synchronization quality is the dimension most specifically affected by real-time constraints.

#### Methodology

**Data Sources:** Maksym works with outputs generated by all other team members, making this a meta-analysis across experimental conditions.

**Metric Battery:**

| Category | Metrics | Tools |
|----------|---------|-------|
| Pixel-level quality | PSNR, SSIM | scikit-image |
| Perceptual quality | LPIPS, FID | torch-lpips, pytorch-fid |
| Identity preservation | CSIM (ArcFace cosine similarity) | InsightFace |
| Lip-sync accuracy | SyncNet Confidence, LED | Pre-trained SyncNet |
| Temporal consistency | Optical flow consistency, temporal LPIPS, jitter variance | RAFT/Farneback, torch-lpips |
| Audio quality | CosSim (speaker similarity), WER | ECAPA-TDNN, Whisper |

**Experimental Protocol:**
1. Compute all metrics for every output across all experimental conditions (architectures × optimization levels × quality conditions).
2. Construct a correlation matrix across all metrics.
3. Perform PCA to identify orthogonal quality dimensions.
4. Conduct sensitivity analysis: which metrics change most under each degradation type (latency increase, FPS reduction, quantization, architecture switch)?
5. Propose a minimum sufficient metric set based on PCA loadings and sensitivity analysis.

**Statistical Analysis:**
- Correlation matrix with significance testing (Pearson r, p-values) for all metric pairs.
- Principal Component Analysis (PCA): identify the number of independent quality dimensions and which metrics load onto each.
- Sensitivity analysis: effect sizes (Cohen's d) for each metric under each degradation condition.
- Redundancy analysis: if two metrics correlate r > 0.85, flag the less sensitive one as redundant.
- Validation: compare recommended metric set against Danil's MOS data to confirm perceptual relevance.

#### Deliverables

- Individual proposal, EDA notebook, and four-section research notebook.
- Comprehensive correlation matrix heatmap (all metrics × all metrics).
- PCA biplot showing metric loadings on quality dimensions.
- Sensitivity analysis table: metric × degradation type → effect size.
- Recommended minimum metric set with justification.
- Shared metrics computation codebase (used by all team members).

#### Unique Contribution

Maksym provides the measurement science foundation. His metric battery is used by everyone; his redundancy analysis prevents the group from over-reporting meaningless numbers; his sensitivity analysis tells each researcher which metrics are most relevant to their specific sub-question. His PCA results become the structural framework for the group's discussion section, showing that the quality-latency tradeoff is not one-dimensional but operates across several orthogonal quality axes.

---

### 🔬 SUB-QUESTION 5: Post-Training Quantization Impact on Voice Conversion Inference

**Researcher:** Aron Wojciechowicz (Infrastructure & Audio Support)

#### Research Question

> "How does Post-Training Quantization (FP32 vs. INT8) impact the trade-off between inference latency, memory footprint, and model accuracy in the audio pipeline of a real-time deepfake system?"

#### What Is Being Investigated

Post-Training Quantization (PTQ) is the most accessible optimization technique for deploying neural networks in resource-constrained environments. It works by reducing the numerical precision of model weights and activations — from 32-bit floating point (FP32, baseline) down to 16-bit (FP16) or 8-bit integers (INT8). This reduction shrinks model size, reduces memory bandwidth requirements, and enables hardware-accelerated low-precision computation, all of which decrease inference latency. However, lower precision means less numerical resolution, which can introduce rounding errors that manifest as quality degradation — particularly in audio generation where waveform fidelity matters.

The core question is: how much latency and memory can be saved through quantization, and at what cost to audio quality? This question is asked specifically about the audio pipeline components (voice conversion encoder, feature extractors, and vocoder) because these components have received far less optimization attention than visual models in the deepfake literature.

The independent variable is quantization precision (three levels: FP32 baseline, FP16, INT8). The dependent variables are inference latency (ms per audio chunk), GPU memory footprint (MB), model file size (MB), and audio quality metrics (MCD — Mel-Cepstral Distortion, CosSim — cosine similarity, WER — word error rate, SNR — signal-to-noise ratio).

#### Why This Matters

Three reasons make this sub-question essential to the group's research.

First, **practical deployment requires optimization**. The other sub-questions characterize the quality-latency tradeoff as it exists with default (unoptimized) models. But real-world deployment almost always involves some form of optimization. Quantization is the most common first step because it requires no retraining, no architecture changes, and no specialized hardware — it can be applied to any PyTorch model in a few lines of code. Understanding what quantization actually costs in quality terms is essential for anyone who wants to deploy a real-time deepfake system (or defend against one).

Second, **the audio pipeline is the optimization blind spot**. The overwhelming majority of deepfake optimization research focuses on visual models (TensorRT for face generation, ONNX Runtime for face detection). Audio models are typically treated as lightweight enough to not need optimization. But as real-time systems push toward lower total latency, every component matters. If the audio encoder contributes 30ms of a 200ms budget, quantizing it to 15ms frees that budget for visual quality improvements. This research quantifies whether that tradeoff is worthwhile.

Third, **vocoder sensitivity is an open question**. Prior literature on speech synthesis optimization (MeloTTS, Wav2Vec2 quantization) suggests that vocoders — the components that convert spectral features back to audio waveforms — are significantly more sensitive to quantization than encoders. This sensitivity differential has not been tested for voice conversion vocoders specifically. If confirmed, it would mean that blanket quantization (quantize everything to INT8) is suboptimal; a mixed-precision approach (INT8 encoder + FP16 vocoder) might be better.

#### Hypotheses

**H5.1 (FP16 Sweet Spot):** FP16 quantization reduces inference latency by 30–40% and memory footprint by approximately 50%, while causing less than 5% degradation in CosSim and less than 0.5 dB increase in MCD compared to FP32. This makes FP16 the optimal default optimization for audio pipelines.

**H5.2 (INT8 Quality Penalty):** INT8 quantization reduces latency by 50–60% and memory by approximately 75%, but causes statistically significant quality degradation (>10% CosSim reduction, >1.0 dB MCD increase), particularly in vocoder output where waveform precision matters. This makes INT8 unsuitable for quality-sensitive applications without additional mitigation.

**H5.3 (Differential Component Sensitivity):** Different pipeline components show different sensitivity to quantization. Specifically, the vocoder (waveform generation) exhibits significantly higher quality degradation under INT8 quantization than the encoder (feature extraction), supporting a mixed-precision deployment strategy.

**H5.4 (Diminishing Returns):** The quality-per-latency ratio decreases with more aggressive quantization — FP32→FP16 provides a better quality-latency tradeoff (more latency saved per unit of quality lost) than FP16→INT8.

#### Methodology

**System Under Test:** The RVC audio pipeline, consisting of:
- **Encoder:** HuBERT/ContentVec feature extractor (produces audio embeddings).
- **Voice Conversion Core:** The trained RVC model that maps source features to target voice characteristics.
- **Vocoder:** NSF-HiFiGAN (converts spectral features to audio waveform).

Each component will be quantized independently and in combination to test differential sensitivity.

**Quantization Framework:** PyTorch native quantization (`torch.quantization`) for static PTQ, with OpenVINO as a secondary framework if PyTorch quantization proves insufficient for INT8 on audio models. TorchAO will be evaluated as an alternative if it provides easier integration.

**Dataset:** Same 100 FakeAVCeleb audio files used by Alex, ensuring results are directly comparable.

**Experimental Protocol:**

**Phase 1 — Baseline Profiling (FP32):**
1. Run the full RVC pipeline on all 100 audio files in FP32 (default precision).
2. Record per-component latency, total latency, memory usage, and model sizes.
3. Compute all quality metrics on outputs.
4. This establishes the unoptimized baseline.

**Phase 2 — FP16 Quantization:**
1. Convert all pipeline components to FP16 using `model.half()`.
2. Re-run the full pipeline on the same 100 files.
3. Record the same latency, memory, and quality metrics.
4. Compare against FP32 baseline.

**Phase 3 — INT8 Quantization:**
1. Apply static post-training quantization to INT8 using PyTorch's quantization API (calibrate on a small subset of audio files).
2. Re-run the full pipeline.
3. Record all metrics.
4. Compare against both FP32 and FP16.

**Phase 4 — Component-Level Sensitivity Analysis:**
1. Quantize only the encoder to INT8 (keep vocoder at FP32) → measure quality.
2. Quantize only the vocoder to INT8 (keep encoder at FP32) → measure quality.
3. Compare: which component is more sensitive?
4. Test mixed precision: INT8 encoder + FP16 vocoder → measure quality and latency.

**Phase 5 — Statistical Analysis and Tradeoff Curves:**
1. Plot latency vs. quality for all precision levels.
2. Plot memory vs. quality for all precision levels.
3. Compute cost-benefit ratios (latency saved per unit quality lost).
4. Test all hypotheses with appropriate statistical tests.

**Measurements:**

| Metric | Tool | What It Captures |
|--------|------|-----------------|
| **Inference Latency** (ms/chunk) | `time.perf_counter()`, PyTorch profiler | Processing speed |
| **GPU Memory** (MB) | `torch.cuda.max_memory_allocated()`, nvidia-smi | Memory footprint |
| **Model Size** (MB) | File system `os.path.getsize()` | Storage/deployment cost |
| **MCD** (Mel-Cepstral Distortion, dB) | librosa + custom computation | Spectral quality — detects quantization noise in frequency domain |
| **CosSim** (Cosine Similarity) | ECAPA-TDNN (SpeechBrain) | Speaker identity preservation |
| **WER** (Word Error Rate, %) | Whisper ASR | Speech intelligibility |
| **SNR** (Signal-to-Noise Ratio, dB) | scipy | Overall audio quality |
| **PESQ** (Perceptual Evaluation of Speech Quality) | pesq library | Standardized perceptual audio quality (if feasible) |

**Why MCD is the primary quality metric:** Mel-Cepstral Distortion operates in the frequency domain and is specifically designed to detect spectral differences between two audio signals. Quantization introduces rounding errors that manifest as spectral noise — subtle changes in frequency content that may not affect CosSim (which measures overall speaker embedding similarity) but are audible as buzzing, hissing, or metallic timbre. MCD detects exactly this type of degradation and is therefore the most sensitive metric for quantization impact assessment.

**Statistical Analysis:**
- Repeated-measures ANOVA (3 precision levels: FP32, FP16, INT8) for each metric.
- Paired t-tests with Bonferroni correction for pairwise precision comparisons.
- Two-way ANOVA (precision level × pipeline component) for the sensitivity analysis.
- Effect sizes: Cohen's d for each comparison, η² for ANOVA.
- Cost-benefit analysis: ΔLatency / ΔQuality ratio for each quantization step.
- Significance level: α = 0.05.

#### Deliverables

- `aron_wojciechowicz_proposal.pdf` — Research proposal with PTQ literature review, RVC baseline definition, experimental design.
- `aron_wojciechowicz_eda.ipynb` — EDA of the audio dataset, FP32 baseline profiling results, initial quality measurements.
- `aron_wojciechowicz_research_notebook.ipynb` — Four sections:
  - Section 1: PTQ implementation for RVC pipeline, environment documentation, deviations from proposal.
  - Section 2: Quantitative results across all precision levels, tables, and visualizations.
  - Section 3: Hypothesis testing, baseline comparison, component sensitivity analysis.
  - Section 4: Conclusions, limitations (single GPU, single model architecture), future directions (QAT, knowledge distillation, TensorRT).
- Visualization: Latency-vs-quality tradeoff curves per precision level, memory reduction bar charts, component sensitivity comparison, mixed-precision recommendation diagram.

#### Unique Contribution

Aron is the only researcher examining optimization techniques, and specifically the only one examining audio-side optimization. His findings directly inform deployment recommendations: given Filipp's architecture comparison and Alex's chunk-length analysis, Aron's quantization results complete the picture by showing how much further the latency can be pushed through model optimization. The component-level sensitivity analysis provides particularly novel insight — if vocoders are confirmed to be quantization-sensitive, this has implications beyond this specific project for anyone deploying real-time speech synthesis or voice conversion.

---

## 3. HOW DO WE MEASURE IT? (Measurement Framework)

The measurement framework is organized into six categories. All researchers share a common metrics codebase (maintained by Maksym) to ensure consistency.

### 3.1 System Performance Metrics

| Metric | Definition | Unit | Measurement Tool | Real-Time Threshold |
|--------|-----------|------|------------------|---------------------|
| End-to-End Latency | Time from audio input to video frame output | milliseconds (ms) | Python `time.perf_counter()` | <200ms |
| Per-Frame Latency | Time to process a single video frame | ms/frame | PyTorch profiler | <33ms (for 30 FPS) |
| Per-Chunk Audio Latency | Time to process one audio buffer | ms/chunk | Python `time.perf_counter()` | <50ms |
| Throughput | Frames processed per second | FPS | Frame count / wall time | ≥25 FPS |
| Component Latency | Time per pipeline stage | ms | Stage-wise profiling wrappers | — |
| GPU Memory Usage | Peak VRAM allocation | GB / MB | `nvidia-smi`, `torch.cuda.max_memory_allocated()` | — |
| Model Size | Stored model file size | MB | `os.path.getsize()` | — |

### 3.2 Audio-Visual Synchronization Metrics (Primary Quality Dimension)

| Metric | Definition | Tool | Range | Threshold |
|--------|-----------|------|-------|-----------|
| Lip Error Distance (LED) | Temporal offset between audio and lip motion | Audio-visual alignment analysis | 0–500ms | <40ms excellent, <80ms acceptable, >120ms noticeable |
| SyncNet Confidence | Pre-trained synchronization model output | SyncNet | 0–10+ (higher = better sync) | >4.0 acceptable |
| Phoneme-Viseme Alignment | Timing correspondence between speech sounds and mouth shapes | Forced alignment + video analysis | 0–1 | >0.7 acceptable |

### 3.3 Visual Quality Metrics

| Metric | Definition | Tool | Range | Notes |
|--------|-----------|------|-------|-------|
| PSNR | Peak Signal-to-Noise Ratio | scikit-image | 20–50 dB (higher = better) | Pixel-level, sensitive to compression |
| SSIM | Structural Similarity Index | scikit-image | 0–1 (higher = better) | Perceptual structure, widely used |
| LPIPS | Learned Perceptual Image Patch Similarity | torch-lpips | 0–1 (lower = better) | Deep feature comparison |
| FID | Fréchet Inception Distance | pytorch-fid | 0–500 (lower = better) | Distributional quality |
| CSIM | Cosine Similarity of face embeddings | ArcFace / InsightFace | 0–1 (higher = better) | Identity preservation |

### 3.4 Audio Quality Metrics

| Metric | Definition | Tool | Range | Notes |
|--------|-----------|------|-------|-------|
| CosSim (Speaker) | Cosine similarity of speaker embeddings | ECAPA-TDNN (SpeechBrain) | 0–1 (higher = better) | Voice identity preservation |
| MCD | Mel-Cepstral Distortion | librosa + custom | 0–20 dB (lower = better) | Spectral quality; primary metric for quantization |
| WER | Word Error Rate | Whisper ASR | 0–100% (lower = better) | Speech intelligibility |
| SNR | Signal-to-Noise Ratio | scipy | dB (higher = better) | Overall audio fidelity |
| PESQ | Perceptual Evaluation of Speech Quality | pesq library | 1–5 (higher = better) | ITU standard, if feasible |

### 3.5 Temporal Consistency Metrics

| Metric | Definition | Method | Range |
|--------|-----------|--------|-------|
| Optical Flow Consistency | Frame-to-frame motion coherence | RAFT or Farneback | 0–1 (higher = more consistent) |
| Temporal LPIPS | Perceptual similarity between consecutive frames | torch-lpips on frame pairs | 0–1 (lower = more consistent) |
| Jitter Variance | Frame-to-frame pixel-level variance in face region | NumPy standard deviation | Lower = better |

### 3.6 Subjective Quality Metrics (Danil's MOS Study Only)

| Metric | Scale | Measurement |
|--------|-------|-------------|
| MOS (Mean Opinion Score) | 1–5 Likert | Average of human ratings across participants |
| Acceptability Rate | 0–100% | Percentage of participants rating ≥3 |
| Detection Rate | 0–100% | Percentage correctly identifying content as AI-generated |
| Inter-Rater Reliability | Cronbach's α | Agreement among participants (target: α > 0.7) |

---

## 4. RESEARCH METHODOLOGY (OVERALL APPROACH)

### 4.1 Experimental Design

**Type:** Comparative experimental study with quantitative analysis and subjective validation.

**Structure:** Five parallel studies sharing a common dataset, baseline, hardware environment, and metrics framework, with each study manipulating a different independent variable to characterize one dimension of the quality-latency tradeoff space.

**Control Strategy:** Variables are controlled by:
- Using the same dataset (FakeAVCeleb subset) across all experiments.
- Running all experiments on the same hardware (BUas Linux server with 8× NVIDIA L40S GPUs, 46GB VRAM each, CUDA 12.8).
- Using a shared metrics codebase for consistency.
- Establishing a common offline baseline (Wav2Lip) against which all real-time results are compared.

### 4.2 Shared Dataset

**Primary Dataset:** FakeAVCeleb v1.2 (Khalid et al., 2021)

- Publicly available under CC BY 4.0 license.
- Contains approximately 500 real and 20,000 manipulated video clips of celebrities.
- This study uses a curated subset of 100 real videos, selected for: duration 10–30 seconds, clear speech (no prolonged silence), frontal face visible in >70% of frames, diverse demographics and lighting conditions.
- Known limitation: 25–30ms leading silence bias in fake videos (documented, accounted for in analysis).

**Why FakeAVCeleb:** It is the standard benchmark for audio-visual deepfake research. Using it enables direct comparison with published results and ensures our findings are meaningful to the broader research community.

### 4.3 Shared Baseline

**Offline Baseline:** Wav2Lip (Prajwal et al., 2020)

Wav2Lip is the established benchmark for lip-sync generation. It processes videos offline (no real-time constraint), producing high-quality lip-synced output at the cost of significant processing time (minutes per short clip). This serves as the "quality ceiling" — the best quality achievable when latency is not a concern.

All researchers compare their real-time results against Wav2Lip's offline quality to quantify the quality cost of real-time operation.

**Audio Baseline:** RVC (Retrieval-based Voice Conversion) via Applio fork, processing full audio files without chunking. This represents the best achievable voice conversion quality when latency is not constrained.

### 4.4 Hardware Environment

All experiments run on the BUas computing infrastructure to eliminate hardware variability.

**Primary:** Linux servers with 8× NVIDIA L40S GPUs (46GB VRAM each), CUDA 12.8, PyTorch with CUDA support.  
**Secondary (Alex local):** Windows with NVIDIA RTX 5070 (12GB VRAM), CUDA 13.0 — used for setup and pilot testing, with final experiments run on the server for consistency.

**Environment Management:** UV package manager for Python dependency resolution, with project-specific virtual environments to prevent dependency conflicts (documented issue from Applio/Gradio setup).

**Specifications documented in every research notebook:** GPU model and VRAM, CPU specifications, RAM, CUDA version, PyTorch version, Python version, OS. This enables reproducibility.

### 4.5 Statistical Analysis Framework

All sub-questions follow a common statistical methodology, adapted to their specific experimental designs.

**Significance Level:** α = 0.05 for all hypothesis tests.

**Effect Size Reporting:** Cohen's d for pairwise comparisons, η² (eta-squared) for ANOVA, r and r² for correlations. Results are reported with confidence intervals where possible.

**Multiple Comparison Correction:** Bonferroni correction when conducting multiple pairwise tests to control family-wise error rate.

**Common Tests by Sub-Question:**

| Sub-Question | Primary Test | Post-Hoc | Additional |
|--------------|-------------|----------|------------|
| SQ1 (Alex, Chunk Length) | Repeated-measures ANOVA (5 conditions) | Tukey HSD | Segmented regression for saturation |
| SQ2 (Filipp, Architecture) | One-way ANOVA (3 systems) | Bonferroni pairwise | Two-way ANOVA (system × difficulty) |
| SQ3 (Danil, Perception) | Correlation (metrics vs MOS) | — | Logistic regression, ROC analysis |
| SQ4 (Maksym, Metrics) | PCA, correlation matrix | — | Sensitivity analysis (Cohen's d) |
| SQ5 (Aron, PTQ) | Repeated-measures ANOVA (3 precisions) | Paired t-tests (Bonferroni) | Two-way ANOVA (precision × component) |

**Power Analysis:** With 100 test videos and typical deepfake quality metric variance, we expect sufficient statistical power (>0.80) to detect medium effect sizes (d > 0.5). If pilot data suggests lower power, sample size will be increased or the effect size threshold adjusted (documented as a deviation from proposal).

### 4.6 Individual Experimental Protocols

Every researcher follows this five-phase structure (mapped to ILOs):

1. **Setup Phase (ILO 7.3A):** Install required models, verify installation with test data, document environment, note any deviations from proposal.
2. **Pilot Phase:** Process 10 test samples, verify metrics computation, identify issues, calibrate expectations.
3. **Execution Phase (ILO 7.3B):** Process full dataset, save all outputs and metrics, document errors or failures, checkpoint intermediate results.
4. **Analysis Phase (ILO 7.3C):** Statistical hypothesis testing, baseline comparison, visualization, interpretation.
5. **Synthesis Phase (ILO 7.3D):** Conclusions, limitations, future work, integration with group findings.

### 4.7 Reproducibility Standards

Following best practices from CVPR and NeurIPS reproducibility guidelines:

- All code committed to Git with version control.
- Random seeds set and documented for any stochastic processes.
- Intermediate results saved to disk (checkpointing) so analyses can be re-run without recomputing inference.
- Environment specifications recorded (Python version, library versions, hardware).
- Configuration files for all experiments committed alongside code.

---

## 5. WHY IS THIS RESEARCH IMPORTANT?

### 5.1 Scientific Contribution

**Fills a critical research gap.** Existing deepfake literature evaluates either offline quality or real-time capability, but not the systematic relationship between the two. Published benchmarks (FaceForensics++, DFDC) assume high-quality offline generation. Real-time system papers (MuseTalk, LivePortrait) report quality metrics but do not compare against offline baselines under controlled conditions. Our study is the first systematic quantification of the quality-latency tradeoff, providing the empirical foundation that future work in this space requires.

**Multi-dimensional analysis.** Most deepfake evaluation studies focus on a single quality dimension (usually visual quality). Our five-pronged approach examines audio quality (Alex), visual architecture tradeoffs (Filipp), perceptual boundaries (Danil), metric validity (Maksym), and optimization impact (Aron) simultaneously. This produces a richer, more complete characterization than any single-metric study could achieve.

**Methodological contribution.** Maksym's metric analysis and Danil's perceptual validation together produce an empirically grounded recommendation for how to evaluate real-time deepfake systems. This evaluation framework — which metrics to use, how they correlate with human perception, and what thresholds indicate acceptable quality — is a reusable contribution to the field.

### 5.2 Practical Impact

**Security and detection.** Understanding the technical capabilities and limitations of real-time deepfakes directly informs defensive strategies. If our research shows that real-time systems consistently produce detectable audio-visual sync errors above 80ms (Danil's threshold), detection systems can specifically target this artifact. If INT8 quantization introduces characteristic spectral noise (Aron's finding), audio-based deepfake detectors can be trained to recognize that signature.

**System design guidance.** Engineers building real-time AV systems (whether for legitimate applications like virtual avatars or for understanding threats) need to make architecture, optimization, and configuration decisions. Filipp's architecture comparison tells them which system to start with. Aron's quantization analysis tells them how to optimize it. Alex's chunk-length analysis sets the audio configuration. Together, these form a practical deployment guide.

**Policy and regulation.** Policymakers assessing the threat of real-time deepfakes need empirical evidence about what is technically achievable. Our perceptual threshold (Danil) establishes whether current real-time systems can fool humans. Our optimization analysis (Aron) shows whether further improvements are easily achievable. This evidence base supports informed regulation.

### 5.3 Societal Relevance

Real-time deepfakes represent an escalating security concern. Industry reports document a 46% year-over-year increase in real-time deepfake attacks in 2024–2025, with $410 million in losses attributed to deepfake fraud in H1 2025 alone. These attacks exploit video conferencing, live streaming, and voice authentication systems — scenarios where real-time generation is essential and where the quality-latency tradeoff directly determines attack viability.

Our research establishes empirical boundaries on this threat: what quality level is achievable at real-time latency, under what conditions humans can detect real-time deepfakes, and how optimization techniques change the threat landscape. This evidence supports the broader societal need for transparency about AI capabilities.

---

## 6. HYPOTHETICAL PAPER STRUCTURE

### Title

"Quantifying the Quality-Latency Tradeoff in Real-Time Audio-Visual Deepfake Generation: A Multi-Dimensional Empirical Study"

### Abstract (~250 words)

The abstract would summarize: the problem (real-time deepfakes must balance quality and latency, but this tradeoff is unquantified), the approach (five complementary analyses using FakeAVCeleb dataset and multiple architectures), key findings (specific numbers from each sub-question — latency thresholds, optimal configurations, perceptual boundaries, recommended metrics), and contribution (first systematic characterization, open-source evaluation framework).

### Paper Sections

**Section 1 — Introduction (2 pages).** Motivation: real-time deepfakes as an emerging threat. Research gap: no systematic quality-latency characterization exists. Research questions: main group question plus five sub-questions. Contributions: first multi-dimensional analysis, perceptual validation, practical guidelines. Paper organization.

**Section 2 — Related Work & Background (3 pages).** Deepfake generation methods: offline (Wav2Lip, VideoRetalking) vs. real-time (MuseTalk, LivePortrait). Voice conversion: RVC, Seed-VC, RT-VC architectures. Quality metrics: visual, audio, synchronization, temporal. Optimization techniques: quantization, TensorRT, pruning. Research gap synthesis: why no existing work addresses the systematic tradeoff.

**Section 3 — Methodology (4 pages).** Experimental design, dataset description and selection criteria, systems evaluated (Wav2Lip baseline, MuseTalk, LivePortrait, RVC), evaluation metrics and measurement protocol, hardware environment, statistical analysis framework, ethics and data management.

**Section 4 — Audio Chunk-Length Analysis (2 pages).** Alex's results: CosSim and latency vs. chunk length, saturation threshold identification, ANOVA results, optimal chunk-length recommendation.

**Section 5 — Architecture Comparison (3 pages).** Filipp's results: three-system comparison, component-level profiling, Pareto frontier, bottleneck identification.

**Section 6 — Optimization via Quantization (2 pages).** Aron's results: FP32 vs. FP16 vs. INT8 latency/quality tradeoffs, component sensitivity analysis, mixed-precision recommendation.

**Section 7 — Metrics Analysis (2 pages).** Maksym's results: metric correlation matrix, PCA quality dimensions, sensitivity analysis, recommended minimum metric set.

**Section 8 — Perceptual Validation (3 pages).** Danil's results: MOS scores across conditions, objective-subjective correlation, acceptability threshold, metric-to-perception mapping.

**Section 9 — Discussion (3 pages).** Integrated findings: what the combined results tell us about the quality-latency tradeoff space. Implications for security (detection targeting), system design (deployment recommendations), and policy (capability assessment). Limitations: single dataset, limited architectures, specific hardware, talking-head only. Threats to validity.

**Section 10 — Conclusions & Future Work (2 pages).** Summary of findings, answers to each research question, future directions (adaptive quality-latency systems, cross-dataset validation, full-body extension, adversarial robustness).

**References (~30–40 citations).** Key papers: MuseTalk, Wav2Lip, LivePortrait, RVC, FakeAVCeleb, SyncNet, ECAPA-TDNN, quantization literature.

---

## 7. SUCCESS CRITERIA

### 7.1 Individual Success (Each Researcher)

**Completion requirements (all ILOs evidenced):**
- Research proposal submitted with literature review, baseline definition, methodology, and hypotheses (ILO 4.4A, 4.4B, 4.4C, 11.2).
- EDA notebook with dataset exploration and baseline results (ILO 4.4B).
- Research notebook with four sections: implementation, results, analysis, synthesis (ILO 7.3A–D).
- Leadership role fully documented in Trello card (ILO 2.5A, B, C).
- Risk log maintained throughout block (ILO 1.5B).
- Ethics application and data management plan completed (ILO 3.5A, B).
- Presentation contribution delivered (ILO 10.2).
- Statistics exam passed (ILO 0.3).

**Quality requirements:**
- Research question is specific, measurable, and directly connected to the group's main question.
- At least three testable hypotheses, each with a clear predicted direction and magnitude.
- Statistical analysis uses appropriate tests with correct assumptions verified.
- Results are interpreted with nuance — unexpected findings are explained, not hidden.
- Limitations are honestly acknowledged and their impact on conclusions assessed.
- Code is structured, commented, and reproducible with checkpointed intermediate results.

### 7.2 Group Success

**Integration:** The five sub-studies complement each other and synthesize into a coherent narrative. The group presentation tells a unified story: "Here is what we investigated, here is what each of us found, and here is what it means collectively."

**Collective insight exceeds individual findings:** The combination of Alex's audio analysis, Filipp's visual architecture comparison, Aron's optimization results, Maksym's metrics framework, and Danil's perceptual validation produces conclusions that no single sub-study could reach alone — specifically, a complete map of the quality-latency tradeoff space with both technical and perceptual grounding.

### 7.3 Medal Opportunities

🥇 **Gold:** After the block ends, draft a manuscript targeting a relevant venue (ACM Multimedia, INTERSPEECH, or a workshop at CVPR/ICCV). Gold is awarded if the manuscript passes desk review (not immediately rejected, assigned to reviewers).

🥈 **Silver:** Achieve the top score for the final presentation among all groups.

🥉 **Bronze:** Score at least 90 points on the statistics exam in Week 7.

---

## 8. RISK MANAGEMENT & MITIGATION

### 8.1 Technical Risks

| Risk ID | Description | Category | Probability | Impact | Score | Priority | Response Strategy | Owner | Monitoring |
|---------|-------------|----------|-------------|--------|-------|----------|-------------------|-------|------------|
| R001 | Model setup failure — MuseTalk, LivePortrait, or RVC fail to install or run on BUas server due to dependency conflicts | Technical | Medium (2) | High (3) | 6 | Critical | **Mitigation:** Start setup Week 1, use Docker containers for isolation, document working configurations. **Contingency:** Fall back to local machines (Alex RTX 5070) for affected components. | Aron (Infra) | Test installations by end of Sprint 2 |
| R002 | GPU server access issues — university proxy, Coder environment limitations, or shared resource contention | Technical | Medium (2) | Medium (2) | 4 | Moderate | **Mitigation:** Reserve GPU time in advance, batch experiments for off-peak hours. **Contingency:** Use local hardware for non-GPU-intensive tasks; move final experiments to server only. | Aron | Weekly server access verification |
| R003 | PTQ quantization fails on RVC components — PyTorch static quantization may not support all operations in the RVC architecture | Technical | Medium (2) | High (3) | 6 | Critical | **Mitigation:** Test quantization on each component separately in pilot phase. **Contingency:** Switch to ONNX Runtime or OpenVINO quantization if PyTorch fails. If all quantization approaches fail, pivot to FP16-only analysis. | Aron | Pilot test by end of Week 4 |
| R004 | FakeAVCeleb download or access delays | Data | Low (1) | Medium (2) | 2 | Low | **Mitigation:** Download in Week 1, verify integrity. **Contingency:** Dataset is publicly available from multiple mirrors. | Aron (DMP) | Confirm dataset availability Week 1 |
| R005 | Computation time exceeds available budget — full metric computation across all conditions takes longer than planned | Technical | Medium (2) | Medium (2) | 4 | Moderate | **Mitigation:** Profile computation time in pilot phase, reduce dataset size if needed (50 instead of 100 videos). **Contingency:** Prioritize primary metrics, compute secondary metrics only if time permits. | All | Weekly progress review in standups |

### 8.2 Research Risks

| Risk ID | Description | Category | Probability | Impact | Score | Priority | Response Strategy | Owner | Monitoring |
|---------|-------------|----------|-------------|--------|-------|----------|-------------------|-------|------------|
| R006 | No significant quality-latency tradeoff found — real-time systems perform comparably to offline baseline | Research | Low (1) | High (3) | 3 | Moderate | **Acceptance:** Null results are valid research outcomes. Document the finding and discuss implications (real-time systems have matured beyond the quality-latency tradeoff). | All | Preliminary results by Week 5 |
| R007 | Insufficient statistical power — metric variance too high to detect medium effects with 100 videos | Research | Medium (2) | Medium (2) | 4 | Moderate | **Mitigation:** Run power analysis in pilot phase. **Contingency:** Increase sample size, use non-parametric tests, or report effect sizes with wide confidence intervals. | Maksym | Power analysis in EDA phase |
| R008 | MOS study recruitment failure — unable to recruit 20+ participants for perceptual evaluation | Research | Medium (2) | High (3) | 6 | Critical | **Mitigation:** Start recruiting in Week 3, use BUas student networks, keep study short (20–30 min). **Contingency:** Reduce to 15 participants minimum; supplement with online recruitment if approved by ethics. | Danil | Recruitment tracking weekly |
| R009 | Ethics approval delayed — BUas ethics review takes longer than expected, blocking MOS study | Research | Medium (2) | High (3) | 6 | Critical | **Mitigation:** Submit ethics application by end of Week 2. **Contingency:** Design study so that objective analysis is complete even without subjective data; MOS becomes "supplementary" rather than "essential." | Danil, Aron | Ethics submission date tracking |
| R010 | Individual sub-studies don't integrate — five separate findings that don't synthesize into a coherent group narrative | Research | Low (1) | High (3) | 3 | Moderate | **Mitigation:** Regular group sync meetings (standups), shared metrics framework, explicit connections between sub-questions documented in proposals. | Scrum Master | Sprint reviews |

### 8.3 Timeline Risks

| Risk ID | Description | Category | Probability | Impact | Score | Priority | Response Strategy | Owner | Monitoring |
|---------|-------------|----------|-------------|--------|-------|----------|-------------------|-------|------------|
| R011 | Implementation phase delays — setup and debugging takes longer than planned, compressing analysis time | Timeline | Medium (2) | High (3) | 6 | Critical | **Mitigation:** Parallel work — write analysis scripts while waiting for experiment results. Start implementation immediately after proposal submission. | All | Gantt chart weekly review |
| R012 | Statistics exam preparation conflicts with research execution | Timeline | Medium (2) | Medium (2) | 4 | Moderate | **Mitigation:** Schedule statistics study for evenings/self-study days. Complete most experimental work before Week 6 mock exam. | All | Individual time management |
| R013 | Writing and documentation time insufficient — research notebook requires more writing time than planned | Timeline | Medium (2) | Medium (2) | 4 | Moderate | **Mitigation:** Write progressively — document methods as they are implemented, not after. Use notebook markdown cells immediately after each experiment. | All | Sprint reviews |

### 8.4 Data and Ethics Risks

| Risk ID | Description | Category | Probability | Impact | Score | Priority | Response Strategy | Owner | Monitoring |
|---------|-------------|----------|-------------|--------|-------|----------|-------------------|-------|------------|
| R014 | Generated deepfake content leaked or misused | Ethics | Low (1) | High (3) | 3 | Moderate | **Avoidance:** All generated content stays on university server. No local copies of deepfake videos. No public sharing. Deletion within 2 weeks of final grade. | Aron (DMP) | DMP adherence monitoring |
| R015 | GDPR compliance failure — improper handling of biometric data in FakeAVCeleb | Ethics | Low (1) | High (3) | 3 | Moderate | **Mitigation:** DMP completed and adhered to. Data minimization (100 video subset, not full dataset). Lawful basis established (GDPR Art. 9(2)(j) — scientific research). | Aron (DMP) | DMP periodic review |

---

## 9. FINAL SUMMARY

### What Makes This Research Strong

**Clear, focused problem.** The quality-latency tradeoff is well-defined, practically relevant, and measurable. It is not a vague exploratory study but a specific empirical characterization with testable hypotheses and quantitative outcomes.

**Five complementary sub-questions that cover the full tradeoff space.** Alex quantifies the audio dimension, Filipp maps the visual architecture space, Aron measures optimization impact, Maksym validates the measurement framework, and Danil grounds everything in human perception. Each sub-question stands alone as independent research, but together they produce insight that no single study could achieve.

**Feasibility.** All models are pre-trained and publicly available. The dataset (FakeAVCeleb) is a standard benchmark. The hardware is available (BUas server + local GPUs). The timeline is realistic: 3 weeks of proposal/design, 4 weeks of implementation/analysis, 1 week of presentation/submission. No custom training is required for baseline experiments.

**Rigorous methodology.** Common dataset, common baseline, common metrics, shared hardware, and a statistical framework with appropriate tests, effect sizes, and multiple comparison corrections. Reproducibility standards follow CVPR/NeurIPS guidelines.

**Relevance.** Real-time deepfakes are a 2025–2026 frontier threat with documented financial and security impact. This is not historical analysis but current, urgent research that informs detection, policy, and system design decisions being made right now.

### What Each Researcher Contributes to the Main Question

| Researcher | Sub-Question Focus | Answers | Key Deliverable |
|------------|-------------------|---------|-----------------|
| **Alex** | Audio chunk-length | "What is the minimum audio buffer for acceptable voice quality?" | Saturation threshold + optimal chunk length |
| **Filipp** | Visual architecture | "Which system design produces the best quality-latency tradeoff?" | Pareto frontier + architecture ranking |
| **Danil** | Perceptual quality | "When is quality good enough to fool (or not fool) humans?" | Acceptability threshold + metric-perception correlation |
| **Maksym** | Evaluation metrics | "Which metrics actually capture meaningful quality differences?" | Minimum metric set + sensitivity analysis |
| **Aron** | PTQ optimization | "How much latency can be saved through quantization, and at what quality cost?" | Mixed-precision recommendation + component sensitivity |

**Together:** A complete empirical map of the quality-latency tradeoff in real-time audio-visual deepfake generation, grounded in both objective metrics and human perception, with actionable deployment and detection recommendations.

### Immediate Next Steps (Week 3 — Proposal Deadline Friday Feb 27)

**For the entire group:**
1. Finalize individual research proposals based on this plan.
2. Run pilot experiments (10 clips each) to populate baseline results tables in proposals.
3. Create EDA notebooks with dataset exploration and preliminary measurements.
4. Ensure ethics application and data management plans are complete (ILO 3.5).
5. Update risk logs with current risks and mitigation actions (ILO 1.5B).
6. Submit proposals to Brightspace by Friday Feb 27 at 17:00.

**For Aron specifically:**
1. Finalize PTQ proposal with literature review on audio model quantization (OpenVINO MeloTTS, Wav2Vec2 quantization precedents).
2. Run FP32 baseline profiling on RVC pipeline — measure latency, memory, and quality metrics for 10 pilot clips.
3. Document FP32 baseline results in EDA notebook and proposal Appendix.
4. Complete Scrum Master Sprint 2 documentation (ILO 2.5 card on Trello).
5. Verify PTQ implementation feasibility — test `torch.quantization` on a simple RVC component to confirm it works before proposing it in the methodology.

---

**Document Version:** 2.0  
**Last Updated:** February 25, 2026  
**Status:** Draft for team review and proposal finalization  
**Next Review:** Sprint 2 Review (Week 4)
