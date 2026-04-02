# FAIR Checklist
**Date:** 2026-02-23 
**Group:** O1  
**Project:** Real-Time Audio-Visual Deepfake Generation: Quality-Latency Tradeoff Analysis  
**Institution:** Breda University of Applied Sciences  
**Year:** 2025–2026  
**Version:** v1.0  
**FAIR Principles Source:** https://www.go-fair.org/fair-principles/

---

## Findable

### F1. Data are assigned a globally unique and persistent identifier.

- **FakeAVCeleb dataset:** Identified by its original publication DOI (Khalid et al., 2021, NeurIPS Datasets and Benchmarks Track) and the official repository URL. The dataset version used is v1.2.
- **Pre-trained model weights:** Identified by their public repository URLs and commit hashes (MuseTalk, Wav2Lip, InsightFace on GitHub/HuggingFace). Specific versions are documented in the research notebook.
- **Generated outputs (deepfake videos, quantised model variants):** These are ephemeral research artefacts and are not assigned persistent identifiers. They are regenerable from the documented pipeline and are deleted after metric computation.
- **Performance metrics and logs:** Stored in the project's private GitHub repository with commit-based versioning. No external persistent identifiers are assigned, as these are internal research artefacts.

### F2. Data are described with rich metadata.

- All experiments are logged with structured metadata including: model name, precision level (FP32/FP16/INT8), input video identifier, timestamp, hardware configuration (GPU model, CUDA version), software versions (PyTorch, ONNX Runtime, OpenVINO), and quantisation parameters.
- Metric CSV files include column headers with units and descriptions.
- The research notebook documents experimental conditions for each run, enabling interpretation of results in context.

### F3. Metadata clearly and explicitly include the identifier of the data they describe.

- Each metric log entry references the corresponding input video by its FakeAVCeleb filename and the model configuration used (e.g., `musetalk_fp32`, `rvc_int8_openvino`).
- Annotation and configuration files use consistent naming: `[system]_[precision]_[experiment_id].[ext]`.

### F4. Data are registered or indexed in a searchable resource.

- **FakeAVCeleb:** Indexed in academic databases and accessible via its official repository.
- **Code and configurations:** Stored in the group's private GitHub repository within the BUas GitHub Organisation, searchable by team members and assessors.
- **Experimental data:** Documented in the research notebook, which serves as the searchable index for all experimental outputs.

---

## Accessible

### A1. Data are retrievable by their identifier using a standardised communication protocol.

- **FakeAVCeleb:** Downloadable via HTTPS from the official repository.
- **Code and notebooks:** Accessible via Git (HTTPS) from the BUas GitHub Organisation.
- **Experimental outputs:** Accessible via SSH from the BUas Linux server during the project duration.

### A1.1. The protocol is open, free, and universally implementable.

- All retrieval uses open protocols: HTTPS for dataset and code access, SSH for server access. No proprietary protocols are required.

### A1.2. The protocol allows for an authentication and authorisation procedure where necessary.

- **Server access:** Requires SSH key authentication. Access restricted to the five cohort members and the assigned mentor.
- **GitHub repository:** Requires authenticated BUas GitHub Organisation account with repository-level permissions.
- **FakeAVCeleb:** Publicly accessible under CC BY 4.0; no authentication required for download.

### A2. Metadata are accessible, even when the data are no longer available.

- Even after generated outputs are deleted post-project, the research notebook and proposal document all experimental conditions, configurations, metric definitions, and results. These documents are retained as academic deliverables.
- The FakeAVCeleb dataset remains available from its original source independently of this project.
- Model weights remain available from their respective public repositories.

---

## Interoperable

### I1. Data use a formal, accessible, shared, and broadly applicable language for knowledge representation.

- **Video data:** MP4 (H.264 codec), a universal standard.
- **Audio data:** WAV (PCM 16-bit, 16kHz), an uncompressed standard format.
- **Metrics:** CSV and JSON, widely supported tabular and structured data formats.
- **Code:** Python (.py, .ipynb), the standard language for AI/ML research.
- **Documentation:** Markdown (.md), a widely supported plain-text format.

### I2. Data use vocabularies that follow FAIR principles.

- Metric names follow conventions from the deepfake and audio-visual research literature: SSIM, PSNR, FID, LPIPS (visual quality); SECS, MCD, WER (audio quality); LSE-D, LSE-C (lip-sync); MOS (perceptual quality).
- Quantisation terminology follows standard ML conventions: FP32 (single precision), FP16 (half precision), INT8 (8-bit integer), PTQ (Post-Training Quantisation).
- These terms are defined in the research proposal and notebook to ensure consistent interpretation.

### I3. Data include qualified references to other data.

- Each experimental run references: (1) the specific FakeAVCeleb video identifiers used as input, (2) the model and version, (3) the quantisation configuration, and (4) the hardware environment.
- The research notebook cross-references the proposal for methodology justification and the DMP for data handling procedures.

---

## Reusable

### R1. Data are richly described with a plurality of accurate and relevant attributes.

- Metric files include: experiment ID, model configuration, input video ID, all computed metric values with units, timestamp, and hardware specification.
- The research notebook provides written interpretation of results alongside raw data, supporting contextual understanding.

### R1.1. Data are released with a clear and accessible data usage licence.

- **FakeAVCeleb:** Released under the Creative Commons Attribution 4.0 International License (CC BY 4.0), permitting sharing and adaptation with attribution.
- **Pre-trained model weights:** Used under their respective open-source licences (MIT, Apache 2.0, or as specified by the original authors).
- **Project code:** Stored in a private BUas GitHub repository. No public licence is assigned during the project. Code may be made available for academic purposes after completion, subject to mentor approval.
- **Generated deepfake outputs:** Not released publicly. This is an intentional ethical decision to prevent misuse.

### R1.2. Data are associated with detailed provenance.

- The research notebook records: which dataset version was used, which model checkpoints were loaded, which quantisation settings were applied, which hardware was used, and which software versions were installed.
- All code is version-controlled via Git with commit history serving as provenance records.

### R1.3. Data meet domain-relevant community standards.

- Evaluation follows established deepfake benchmarking practices from CVPR, ICCV, and ACM Multimedia publications.
- Metrics are computed using standard open-source implementations (scikit-image for SSIM/PSNR, torch-lpips for LPIPS, SpeechBrain for speaker embeddings, Whisper for WER).
- Experimental design follows controlled comparison methodology standard in ML research (same dataset, same hardware, varying one independent variable).

---

## Limitations of FAIR Compliance

1. **Generated deepfake outputs are not shared publicly.** This limits reusability but is an intentional ethical decision. Publishing optimised deepfake generation outputs could enable misuse.
2. **Quantised model variants are ephemeral.** INT8/FP16 quantised models are created during experiments and deleted after analysis. They are not published as standalone artefacts but are fully reproducible from documented configurations.
3. **Model weights are not redistributed.** Pre-trained weights are used under their respective licences but are not redistributed or modified for public use.
4. **Performance metric data is retained only as academic deliverables.** Raw CSV/JSON metrics are not deposited in external repositories but are documented in the research notebook, which is submitted as a graded deliverable.

---

## References

- Khalid, H., et al. (2021). FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset. *NeurIPS Datasets and Benchmarks Track*.
- Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data*, 3, 160018.
- GO FAIR Initiative. (2024). FAIR Principles. https://www.go-fair.org/fair-principles/
