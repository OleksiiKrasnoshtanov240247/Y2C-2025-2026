# Privacy and GDPR Compliance Checklist
**Date:** 2026-02-23
**Group:** O1  
**Project:** Real-Time Audio-Visual Deepfake Generation: Quality-Latency Tradeoff Analysis  
**Institution:** Breda University of Applied Sciences  
**Year:** 2025–2026  
**Version:** v1.0

---

## 1. Do we process personal data?

**Yes — partially.** The FakeAVCeleb dataset (v1.2) contains video and audio recordings of real, identifiable individuals (celebrities). Under GDPR, facial images and voice recordings constitute biometric data, which is classified as special category data under Article 9.

However, this project does not collect new personal data. All biometric data originates from a publicly available research dataset released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). The data subjects are public figures whose likenesses were already publicly available at the time of dataset creation.

**Additionally**, the MOS (Mean Opinion Score) perceptual study involves human participants (30+ adult volunteers). However, only anonymous numerical ratings (1–5 Likert scale) are collected. No names, email addresses, faces, or other identifiable information of participants is stored. The MOS data is therefore not considered personal data under GDPR.

---

## 2. What is the aim of processing data?

By processing the FakeAVCeleb video data through deepfake generation pipelines (MuseTalk, Wav2Lip, RVC) at different quantisation levels (FP32, FP16, INT8), we aim to systematically quantify the quality-latency tradeoff in real-time audio-visual deepfake generation.

**Processed data include:**
- Real celebrity video clips (input to generation pipelines)
- Synthetic deepfake video/audio outputs (generated during experiments)
- Numerical performance metrics (latency, SSIM, PSNR, FID, SECS, MCD, WER, MOS)
- System profiling data (GPU utilisation, memory usage, inference timing)

**Purpose:** Academic research within BUas ADS&AI Block C (Year 2, 2025–2026). The research contributes to understanding real-time deepfake capabilities, informing detection strategies and responsible AI development.

---

## 3. What is the basis for processing?

### a. Do respondents give consent?

**For the FakeAVCeleb dataset:** The data subjects (celebrities) did not provide individual consent for this specific study. However, the dataset was created and published for academic research under CC BY 4.0, with the original authors establishing the ethical basis for its use. Our processing is justified under GDPR Article 9(2)(j) — scientific research with appropriate safeguards.

**For MOS study participants:** Yes. All participants receive written information about the study purpose and procedures before participation. They provide explicit informed consent and may withdraw at any time. Their responses are anonymous — no personal data is collected or stored.

### b. Is there a general interest or a legitimate interest for processing during research?

**Yes.** There is a legitimate scientific interest in understanding the quality-latency tradeoff in real-time deepfake systems. This research:
- Contributes to the academic understanding of AI-generated media capabilities and limitations
- Informs the development of deepfake detection technologies
- Supports responsible AI policy development
- Addresses an emerging security concern (real-time identity manipulation)

The processing is proportional to the research goal: we use only a curated subset of approximately 100 videos (from approximately 20,000 available), which is the minimum necessary for statistically meaningful comparison.

---

## 4. How do we estimate the risk of processing personal data?

### a. Is there a significant risk or a low risk?

**Medium risk.** The risk assessment considers:

| Factor | Risk Level | Justification |
|--------|-----------|---------------|
| Data type | Medium | Biometric data (faces, voices) is special category under GDPR Article 9 |
| Data subjects | Low | Public figures with publicly available likenesses |
| New data collection | Low | No new biometric data is collected; only existing public dataset is used |
| Generated outputs | Medium | Synthetic deepfakes derived from biometric data have dual-use potential |
| MOS participants | Low | Only anonymous numerical ratings collected; no identifiable data |
| Storage security | Low | University server with SSH access and file-level permissions |
| Distribution risk | Low | No generated deepfakes are published or shared externally |

**Overall assessment: MEDIUM.** The primary risk factors are the biometric nature of the source data and the dual-use potential of deepfake technology, not the likelihood of privacy breaches against data subjects.

### b. Do we doubt this risk level?

No. The risk level is agreed within the research cohort and is consistent with the BUas Ethics Review Application, which classifies the project as MEDIUM risk based on biometric data usage (E0.2b2) and dual-use concerns (E0.2b3).

---

## 5. How do we process personal data in accordance with the GDPR?

The following GDPR principles are applied:

### 5.1. Lawfulness, Fairness, and Transparency

- **Lawful basis:** GDPR Article 9(2)(j) — processing of special category data for scientific research purposes with appropriate safeguards.
- **Dataset licence:** FakeAVCeleb is released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), permitting use for any purpose including research with attribution.
- **Transparency:** This Privacy Checklist, the Data Management Plan, and the Ethics Application document all data handling procedures. The research proposal describes data usage.

### 5.2. Purpose Limitation

- Data is processed exclusively for quantifying quality-latency tradeoffs in deepfake generation systems within BUas Block C academic research.
- Generated outputs are used only for metric computation and illustrative examples in the research notebook and presentation.
- No data is repurposed for any other application, commercial or otherwise.

### 5.3. Data Minimisation

- A curated subset of approximately 100 real videos is used rather than the full dataset of approximately 20,000 clips.
- Selection criteria focus on technical requirements (clear speech, frontal face, diverse demographics) rather than collecting maximum possible data.
- Only metrics necessary for the research questions are computed and stored.

### 5.4. Accuracy

- The FakeAVCeleb dataset is a peer-reviewed benchmark published at NeurIPS. Its quality and accuracy have been validated by the research community.
- Metrics are computed using established, open-source implementations to ensure accuracy and reproducibility.

### 5.5. Storage Limitation

- Generated deepfake outputs are stored on the BUas server only during active experimentation and are deleted within 48 hours of metric extraction.
- All project data (including the FakeAVCeleb subset) is deleted from the server within 30 days of final grade publication (estimated: April 2026).
- Anonymous MOS ratings are retained only as part of the academic deliverables (research notebook).

### 5.6. Integrity and Confidentiality

- All data is stored on BUas Linux servers with SSH key authentication (no password-based login).
- File permissions restrict dataset access to the five cohort members and the assigned mentor (chmod 770).
- No data is uploaded to public cloud services, personal storage, or shared externally.
- No generated deepfake videos are downloaded to personal devices.

### 5.7. Accountability

- This Privacy and GDPR Compliance Checklist documents the assessment and compliance measures.
- The Data Management Plan provides detailed data handling procedures.
- Adherence is monitored through server directory listings, Git commit history, and deletion confirmation logs (documented in the learning log, ILO 3.5A).

---

## 6. Additional Safeguards for Biometric Data

Given the special category nature of biometric data under GDPR Article 9, the following additional safeguards are implemented:

1. **No new identity manipulation:** Generated deepfakes use only faces already present in the FakeAVCeleb dataset. No new individuals' likenesses are synthesised.
2. **No external distribution:** Generated deepfake content is not published, shared on social media, or distributed outside the research cohort and assessors.
3. **Ephemeral outputs:** Synthetic video/audio outputs are treated as intermediate research artefacts, not as data products. They are deleted after metric computation.
4. **Responsible disclosure:** Research findings are framed for defensive applications (detection, policy). No novel attack capabilities or generation pipelines are released.
5. **Dual-use mitigation:** The research benchmarks existing public systems; it does not develop new deepfake generation capabilities.

---

## 7. MOS Study — Participant Data Protection

| Aspect | Implementation |
|--------|----------------|
| Data collected | Anonymous numerical ratings (1–5 Likert scale) on perceived naturalness and audio-visual synchronisation |
| Personal data stored | None. No names, emails, IP addresses, or other identifiers are collected |
| Informed consent | Written information sheet provided; explicit consent obtained before participation |
| Right to withdraw | Participants may stop at any time without consequence |
| Data format | Anonymous CSV file containing only numerical scores |
| Storage | BUas Linux server, accessible only to cohort members |
| Retention | Retained as part of academic deliverables; no external publication of raw data |
| Method | Blind A/B testing of generated video artefacts; strictly anonymous |
| Participants | 30+ adult volunteers (BUas students and/or faculty) |

---

## References

- European Parliament and Council. (2016). Regulation (EU) 2016/679 (General Data Protection Regulation). *Official Journal of the European Union*.
- Khalid, H., et al. (2021). FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset. *NeurIPS Datasets and Benchmarks Track*.
- Breda University of Applied Sciences. (2025). Privacy and GDPR Checklist. Internal documentation.
