# Ethical Considerations and Decisions
## Real-Time Audio-Visual Deepfake Generation: Quality-Latency Tradeoff Analysis

**Group:** O1  
**Institution:** Breda University of Applied Sciences  
**Year:** 2025–2026  
**Version:** v1.0

---

## Introduction

In AI research involving deepfake generation, thorough ethical analysis is essential. This document describes and specifies the ethical research and decisions required for this project. Deepfake technology is inherently dual-use: the same systems that enable virtual avatars and accessible communication tools can also be used for identity fraud, misinformation, and non-consensual media manipulation. This ethical framework addresses data privacy, research integrity, responsible AI development, and societal impact.

---

## Data Privacy and Protection

### GDPR Compliance Assessment

This assessment confirms that the research involves **special category data** under GDPR Article 9. The FakeAVCeleb dataset (v1.2) contains video and audio recordings of real, identifiable individuals (celebrities), constituting biometric data. However, the risk is mitigated by the following factors:

- The dataset is publicly available under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
- Data subjects are public figures whose likenesses are already widely accessible
- No new biometric data is collected from any individual
- Processing is justified under GDPR Article 9(2)(j) for scientific research with appropriate safeguards
- Generated outputs are ephemeral and are not published or distributed

The MOS perceptual study collects only anonymous numerical ratings (1–5 Likert scale) from 30+ adult participants via blind A/B testing. No names, emails, faces, or other identifiable information of respondents is stored. This data is not considered personal data under GDPR.

A detailed assessment is documented in the Privacy and GDPR Compliance Checklist (O1_Privacy_GDPR_Checklist.md).

### Data Anonymisation and Protection Strategies

1. **FakeAVCeleb data:** Used as-is from the original repository. No additional anonymisation is applied as the dataset is designed for academic use with established ethical clearance.
2. **Generated outputs:** Treated as ephemeral research artefacts. Deleted within 48 hours of metric computation. Not stored on personal devices or cloud services.
3. **MOS responses:** Collected anonymously. No linking between responses and participant identities is possible.
4. **Metrics and logs:** Contain no personal data — only numerical performance measurements.

### Data Storage Security

All data is stored on BUas Linux servers with SSH key authentication. Access is restricted to the five cohort members and the assigned mentor via file-level permissions (chmod 770). No data is uploaded to public cloud services or personal storage. Details are specified in the Data Storage Protocol (O1_Data_Storage_Protocol.md).

---

## Research Integrity

This project adheres to the **Netherlands Code of Conduct for Research Integrity (2018)** through the five core principles:

### 1. Honesty

- All data sources, model architectures, and methodologies are transparently documented in the research proposal and notebook.
- Performance metrics are accurately reported, including negative results and unexpected outcomes.
- Quantisation effects are measured objectively using established metrics; results are not selectively reported to favour a particular conclusion.
- The dual-use nature of the research is openly acknowledged rather than obscured.

### 2. Scrupulousness

- Experimental protocols are rigorously defined before execution, with controlled comparisons (same dataset, same hardware, varying one independent variable).
- All evaluation metrics use standard, open-source implementations to prevent measurement bias.
- Statistical analysis follows established methods (paired t-tests, ANOVA) with pre-defined significance levels (α = 0.05).
- Any deviations from the research proposal are documented and justified in the research notebook.

### 3. Transparency

- Data collection, processing, and storage procedures are documented in the Data Management Plan (aron_wojciechowicz_dmp.md).
- Experiment configurations (model versions, quantisation parameters, hardware specifications) are version-controlled in the GitHub repository.
- Limitations and uncertainties are explicitly acknowledged in the research notebook and presentation.
- This ethics documentation package is submitted as part of the project deliverables.

### 4. Independence

- The research is conducted as part of BUas academic curriculum with no external funding or commercial interests.
- Evaluation uses objective, automated metrics alongside subjective human evaluation (MOS) to prevent researcher bias.
- No organisation or external party has control over the research methods, execution, or reporting.
- Results are interpreted based on evidence regardless of whether they support or contradict the hypotheses.

### 5. Responsibility

- All team members are accountable for their individual research contributions and adherence to ethical guidelines.
- The Infrastructure & Audio Support role (Aron Wojciechowicz) is specifically responsible for data management and security compliance.
- Generated deepfake outputs are handled responsibly: not shared externally, not used for purposes beyond metric computation, and deleted after analysis.
- The research is framed to contribute to defensive understanding (detection, policy) rather than offensive capability development.

---

## Informed Consent and Data Sourcing

### Dataset: FakeAVCeleb v1.2

- **Source:** Publicly available benchmark dataset (Khalid et al., 2021, NeurIPS Datasets and Benchmarks Track)
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Content:** Approximately 500 real and 20,000 manipulated video clips of celebrities
- **Usage:** Curated subset of approximately 100 real videos for controlled experiments
- **Ethical basis:** Published for academic benchmarking with clearly established usage rights. The original authors obtained and documented the ethical basis for the dataset's creation and distribution.

### Pre-trained Model Weights

| Model | Source | Licence | Usage |
|-------|--------|---------|-------|
| MuseTalk | GitHub / HuggingFace | Repository licence | Inference only (no retraining) |
| Wav2Lip | GitHub | Repository licence | Baseline inference |
| RVC | GitHub | Repository licence | Voice conversion inference |
| InsightFace / Ghost | GitHub | Repository licence | Face swap inference |

All models are used in inference-only mode with pre-trained weights. No models are retrained, modified for distribution, or packaged for external deployment.

### MOS Study Participants

- **Participants:** 30+ adult volunteers (BUas students and/or faculty)
- **Method:** Blind A/B testing of generated video artefacts
- **Consent:** Written information sheet provided before participation; explicit informed consent obtained
- **Right to withdraw:** Participants may withdraw at any time without consequence
- **Data collected:** Strictly anonymous numerical ratings (1–5 Likert scale) on perceived naturalness and audio-visual synchronisation
- **No identifiers stored:** No names, emails, IP addresses, or other identifying information is collected
- **Ethics approval:** A BUas Ethics Review Application has been submitted for non-binding advice

---

## Dual-Use Risk Assessment

### Identified Risks

Deepfake technology poses inherent dual-use risks:

| Risk | Severity | Likelihood in This Project | Mitigation |
|------|----------|---------------------------|------------|
| Generated content used for identity fraud | High | Very Low | Outputs deleted within 48 hours; no external distribution |
| Research enabling more effective real-time deepfakes | Medium | Low | Benchmarks existing public systems; no novel capabilities developed |
| Pipeline code enabling easy deepfake production | Medium | Low | No generation pipeline packaged for deployment; code in private repository |
| Findings misrepresented to exaggerate threat | Low | Low | Responsible framing; limitations clearly documented |
| Reputational risk to BUas from deepfake association | Low | Low | Defensive research framing; ethics documentation package |

### Mitigation Strategy

1. **No novel capability development.** This research benchmarks and evaluates existing, publicly available systems. No new deepfake generation methods, models, or pipelines are created.
2. **No code or model release.** The generation pipeline, quantised models, and experiment code remain in a private BUas repository. No tools are published that could lower the barrier to deepfake creation.
3. **Ephemeral outputs.** Generated deepfake content is treated as intermediate research artefacts and deleted after metric computation.
4. **Defensive framing.** Research findings are positioned to support detection technology development, security assessment, and policy formation.
5. **Responsible disclosure.** Any unexpected findings regarding system vulnerabilities or capabilities are discussed with the mentor before inclusion in deliverables.

---

## Fairness and Bias Prevention

### Dataset Representation

The FakeAVCeleb dataset includes diverse demographics (ethnicity, gender, age). The curated subset of approximately 100 videos is selected to maintain this diversity, preventing bias in quality evaluation metrics toward specific demographic groups.

### Evaluation Bias Prevention

- Objective metrics (SSIM, PSNR, SECS, MCD, WER) are computed automatically using standardised tools, removing subjective researcher bias.
- The MOS study uses randomised presentation order to prevent order effects.
- Blind A/B testing ensures participants rate quality without knowing the system or configuration used.

---

## FAIR Principles

Compliance with FAIR principles is documented in the separate FAIR Checklist (O1_FAIR_Checklist.md). Key points:

- **Findable:** Dataset identified by DOI; experiments logged with structured metadata.
- **Accessible:** Data retrievable via standard protocols (HTTPS, SSH, Git).
- **Interoperable:** Standard formats (MP4, WAV, CSV, JSON, Python); standard metric vocabulary.
- **Reusable:** Clear licences; detailed provenance; reproducible methodology.

**Intentional limitation:** Generated deepfake outputs are not shared publicly, limiting reusability but preventing misuse.

---

## Risk Management and Mitigation

Research risks are tracked in an individual risk log (risk_log.md) and the group Trello board. The key risks specific to ethical considerations are:

| Risk ID | Description | Priority | Mitigation |
|---------|------------|----------|------------|
| R-AW-06 | Ethical concerns raised by reviewers or mentors | Medium | Comprehensive ethics documentation; transparent dual-use acknowledgement; responsible framing |
| R-AW-07 | Generated outputs inadvertently shared or leaked | Medium | Server-only storage; no personal device copies; ephemeral deletion policy; access controls |

Full risk assessment is documented in risk_log.md with probability-impact scoring.

---

## Data Retention and Stewardship

| Data Type | Retention Period | Purpose |
|-----------|-----------------|---------|
| Generated deepfake outputs | 48 hours after metric extraction | Ephemeral — only for quality measurement |
| FakeAVCeleb subset | Deleted within 30 days of final grade | Available from source for future researchers |
| Model weights | Deleted within 30 days of final grade | Available from source repositories |
| Metrics and logs | Retained as academic deliverables | Reproducibility and grading |
| Code | Retained in GitHub repository | Reproducibility and portfolio |
| MOS responses (anonymous) | Retained as academic deliverables | Reproducibility and grading |

A deletion log is maintained to confirm data removal. Details in the Data Storage Protocol (O1_Data_Storage_Protocol.md).

---

## Institutional Compliance

### Breda University of Applied Sciences (BUas)

- Use of institutional infrastructure: BUas Linux server, BUas GitHub Organisation
- Mentor oversight throughout the research block
- Adherence to BUas Code of Conduct and Educational and Exam Regulations
- Ethics Review Application submitted for non-binding advice
- All sources cited using APA style per BUas requirements, including generative AI tool usage

### Netherlands Code of Conduct for Research Integrity (2018)

The research adheres to all standards for good research practices as specified in the Code, including design (standards 1–8), conduct (standards 9–17), reporting results (standards 18–39), assessment and peer review (standards 40–53), and communication (standards 54–55). See the Appendix of the BUas Ethics Review Application Form for the full standard overview.

---

## Stakeholder Communication

- **Mentor:** Regular feedback through sprint reviews and stand-ups; access to Trello board and GitHub repository.
- **Team members:** Daily communication; shared understanding of ethical obligations documented in team message (team_ethics_message.md).
- **MOS participants:** Written information sheet and informed consent form; clear communication that videos contain AI-generated content.
- **Assessors:** Full ethics documentation package submitted as part of deliverables.

---

## References

- European Parliament and Council. (2016). Regulation (EU) 2016/679 (General Data Protection Regulation). *Official Journal of the European Union*.
- Khalid, H., et al. (2021). FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset. *NeurIPS Datasets and Benchmarks Track*.
- Netherlands Code of Conduct for Research Integrity. (2018). KNAW, NFU, NWO, TO2, VSNU.
- Breda University of Applied Sciences. (2025). Research Ethics Review Application Form v1.2.
