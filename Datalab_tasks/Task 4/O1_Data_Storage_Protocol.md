# Data Storage, Sharing, and Protection Protocol
## Real-Time Audio-Visual Deepfake Generation: Quality-Latency Tradeoff Analysis

**Group:** O1  
**Institution:** Breda University of Applied Sciences  
**Project Duration:** February 2026 – April 2026 (8 weeks, Block C)  
**Year:** 2025–2026  
**Version:** v1.0

---

## 1. Introduction

This protocol outlines the storage, documentation, and security practices for managing research data generated during the real-time deepfake quality-latency tradeoff study. It ensures compliance with the FAIR principles (Findable, Accessible, Interoperable, Reusable), the Netherlands Code of Conduct for Research Integrity (2018), GDPR requirements for biometric data, and BUas institutional requirements for data protection.

The protocol addresses the specific requirements of deepfake research, emphasising responsible handling of biometric source data, secure management of synthetic outputs, and ephemeral storage of generated content to prevent misuse.

---

## 2. Data Types

This project generates and processes the following data types:

| Data Type | Format | Approximate Volume | Retention |
|-----------|--------|-------------------|-----------|
| FakeAVCeleb video subset (~100 clips) | MP4 (H.264, 720p/1080p) | ~5 GB | Deleted after project |
| Extracted audio tracks | WAV (PCM 16-bit, 16kHz) | ~500 MB | Deleted after project |
| Pre-trained model weights (MuseTalk, Wav2Lip, RVC, InsightFace) | PTH, ONNX, OpenVINO IR | ~10 GB | Deleted after project (re-downloadable) |
| Quantised model variants (FP16, INT8, TensorRT) | PTH, ONNX, engine files | ~5 GB | Deleted after metric extraction |
| Generated deepfake video/audio outputs | MP4, WAV, PNG frames | ~20 GB (cumulative) | Deleted within 48 hours of metric extraction |
| Performance metrics and logs | CSV, JSON | <100 MB | Retained in research notebook |
| System profiling data (GPU, memory, timing) | CSV, JSON | <50 MB | Retained in research notebook |
| MOS study responses | CSV (anonymous ratings) | <1 MB | Retained in research notebook |
| Python code and notebooks | .py, .ipynb | <50 MB | Retained in GitHub repository |
| Research proposal and documentation | .pdf, .md | <20 MB | Retained as academic deliverables |

---

## 3. Data Collection Process

### 3.1 Dataset Acquisition

The primary dataset, FakeAVCeleb v1.2, is a publicly available benchmark released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). It is downloaded from the official repository via HTTPS.

A curated subset of approximately 100 real videos is selected based on the following criteria:
- Duration: 10–30 seconds
- Clear speech (no silence >2 seconds)
- Frontal face visible in >70% of frames
- Diverse demographics and lighting conditions

### 3.2 Model Weight Acquisition

Pre-trained model weights are downloaded from their respective public repositories:
- **MuseTalk:** GitHub / HuggingFace
- **Wav2Lip:** GitHub
- **RVC (Retrieval-based Voice Conversion):** GitHub
- **InsightFace / Ghost:** GitHub

Specific versions and commit hashes are documented in the research notebook for reproducibility.

### 3.3 Generated Output Production

Deepfake video and audio outputs are generated during experiments by running the curated video subset through each pipeline at different quantisation levels (FP32, FP16, INT8). These are intermediate research artefacts used exclusively for metric computation.

### 3.4 MOS Study Data Collection

Anonymous perceptual quality ratings are collected from 30+ adult participants using blind A/B testing. Only numerical scores (1–5 Likert scale) on perceived naturalness and audio-visual synchronisation are recorded. No participant identifiers are collected.

---

## 4. Folder Structure

```
/data/Y2C/
├── datasets/
│   └── fakeav_celeb_v12/          # FakeAVCeleb subset (~100 videos)
│       ├── real/                    # Original real videos
│       └── metadata/                # Selection criteria, video list
├── models/
│   ├── musetalk/                    # MuseTalk weights + configs
│   ├── wav2lip/                     # Wav2Lip weights
│   ├── rvc/                         # RVC pipeline weights
│   ├── insightface/                 # InsightFace/Ghost weights
│   └── quantised/                   # FP16, INT8, TensorRT variants
├── outputs/
│   ├── aron/                        # Individual experiment outputs
│   ├── alex/
│   ├── filipp/
│   ├── maksym/
│   └── danil/
├── metrics/
│   ├── latency/                     # Timing measurements
│   ├── quality/                     # SSIM, PSNR, FID, LPIPS
│   ├── audio/                       # SECS, MCD, WER
│   ├── sync/                        # LSE-D, LSE-C
│   └── mos/                         # Perceptual study results
├── profiling/                       # GPU, memory, system logs
└── docs/
    ├── README.md                    # Project overview and setup
    └── environment.yml              # Software dependencies
```

### GitHub Repository Structure
```
buas-adsai/Y2C-O1-deepfake-research/
├── src/                             # Shared utility code
├── notebooks/
│   ├── O1_eda.ipynb
│   ├── O1_research_notebook.ipynb
│   └── [other team member notebooks]
├── configs/                         # Experiment configurations
├── docs/
│   ├── proposals/
│   ├── ethics/
│   └── planning/
└── README.md
```

---

## 5. Version Control

- **Code:** Git version control via BUas GitHub Organisation (private repository). All code changes are tracked with commit messages describing the change.
- **Large files:** Model weights, videos, and generated outputs are excluded from Git via `.gitignore`. These reside only on the BUas Linux server.
- **Configuration files:** Experiment configurations (hyperparameters, quantisation settings) are version-controlled in the repository.
- **Documentation:** Research proposal, DMP, ethics documents, and checklists are version-controlled in the `docs/` directory.

---

## 6. Data Security and Backup

### 6.1 Storage Security

| Measure | Implementation |
|---------|----------------|
| **Server access** | SSH key authentication only. No password-based login. |
| **File permissions** | Dataset directories: `chmod 770` (group read/write/execute, no public access). Individual output directories: owner + group access. |
| **Network** | BUas internal network. SSH connections only from authenticated clients. |
| **Encryption** | Data at rest on server filesystem. SSH for encrypted data in transit. |

### 6.2 Access Control

| Role | Access Level |
|------|-------------|
| Team members (5) | Read/write to all project directories |
| Mentor | Read access to all project directories |
| BUas IT | Server administration (no project-level access by default) |
| External parties | No access |

### 6.3 Backup Strategy

| Data Type | Backup Approach |
|-----------|----------------|
| FakeAVCeleb subset | No backup needed — re-downloadable from source |
| Pre-trained weights | No backup needed — re-downloadable from repositories |
| Quantised variants | No backup — regenerable from documented configuration |
| Generated outputs | No backup — ephemeral, regenerable from pipeline |
| Metrics and logs | GitHub repository (version-controlled) |
| Code and notebooks | GitHub repository (primary), local Git clones (secondary) |

**Rationale:** Given the ephemeral nature of generated outputs and the reproducibility of the pipeline, traditional backup strategies are unnecessary. The emphasis is on reproducibility (documented configurations and code) rather than data duplication.

### 6.4 Sensitive Data Handling

- **No local copies of generated deepfakes.** All generation and storage occurs on the BUas server. Team members do not download deepfake videos to personal devices.
- **No cloud storage.** No research data (especially generated deepfakes) is uploaded to public cloud services, Google Drive, Dropbox, or personal storage.
- **No public sharing.** Generated deepfake content is not shared on social media, messaging platforms, or any external channels.

---

## 7. Data Retention and Deletion

| Phase | Timeline | Action |
|-------|----------|--------|
| **Active research** | Feb–Apr 2026 | All data on BUas server for experiment reproducibility |
| **Post-experiment cleanup** | Ongoing during project | Generated deepfake outputs deleted within 48 hours of metric extraction |
| **Post-submission** | Within 30 days of final grade | FakeAVCeleb subset deleted from server |
| **Post-submission** | Within 30 days of final grade | All model weights deleted from server |
| **Post-submission** | Within 30 days of final grade | All remaining generated outputs deleted |
| **Retained permanently** | Academic deliverables | Research notebook, proposal, metrics (as submitted documents) |
| **Retained permanently** | Code repository | Source code (may be archived or made available for academic purposes) |

### Deletion Verification

A deletion log is maintained confirming:
- Which data was removed
- Date of deletion
- Method of deletion (file system removal)
- Person responsible for deletion

This log is included in the learning log (ILO 3.5A) as evidence of DMP adherence.

---

## 8. Ethical Considerations for Data Storage

### 8.1 Dual-Use Risk Mitigation

Generated deepfake outputs pose dual-use risks. The following storage-specific safeguards are implemented:
- Outputs are stored in individual directories with restricted access
- Ephemeral storage policy ensures outputs are not accumulated
- No pipeline or generation code is packaged for easy external deployment
- Server access logs may be reviewed to verify appropriate usage

### 8.2 Biometric Data Protection

The FakeAVCeleb dataset contains biometric data (faces and voices). Storage safeguards include:
- Dataset stored in a dedicated directory separate from other project data
- No modifications to original video files (processed copies are separate)
- Dataset deleted after the research block concludes
- No redistribution of the dataset beyond the research cohort

---

## 9. Compliance Summary

| Standard | Compliance |
|----------|-----------|
| **FAIR Principles** | Documented in separate FAIR Checklist (O1_FAIR_Checklist.md) |
| **GDPR** | Documented in separate Privacy and GDPR Checklist (O1_Privacy_GDPR_Checklist.md) |
| **Netherlands Code of Conduct for Research Integrity (2018)** | Adhered through honest reporting, transparent documentation, reproducible methodology, and responsible data handling |
| **BUas institutional requirements** | Use of institutional infrastructure (server, GitHub), mentor oversight, academic integrity |
| **CC BY 4.0 (FakeAVCeleb licence)** | Attribution provided in all deliverables; usage within licence terms |

---

## 10. Responsible Contact

| Role | Name | Contact |
|------|------|---------|
| Infrastructure & Data Management | Aron Wojciechowicz | 245205@buas.nl |
| QA & User Study | Danil Sysenko | 244760@buas.nl |
| Mentor | Peiman Barnaghi | barnaghi.p@buas.nl |

---

## References

- Khalid, H., et al. (2021). FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset. *NeurIPS Datasets and Benchmarks Track*.
- European Parliament and Council. (2016). Regulation (EU) 2016/679 (General Data Protection Regulation).
- Netherlands Code of Conduct for Research Integrity. (2018). KNAW, NFU, NWO, TO2, VSNU.
- Breda University of Applied Sciences. (2025). Data Storage Protocol guidelines. Internal documentation.
