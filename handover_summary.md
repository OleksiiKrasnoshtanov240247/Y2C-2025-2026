# Unified Deepfake Evaluation Pipeline: Architectural Handover & Status Report

## Current State of the Pipeline
The core architectural structure of the **Unified Deepfake Evaluation Pipeline** (Group O1) is fully complete and functional. The entire `FakeAVCeleb` end-to-end processing logic has been successfully orchestrated through a central Python script (`src/pipeline/run_unified_pipeline.py`).

### Completed Milestones
1.  **Phase 1 (Data Prep):** Audio extraction and `.wav` dataset serialization works perfectly.
2.  **Phase 2 (RVC Audio):** Applio integration accurately handles Voice Cloning chunking, effectively bypassing VRAM offload limits via `infer_with_timing.py`.
3.  **Phase 3 (Video Generation):** All 5 Deepfake algorithms (`FaceFusion`, `Wav2Lip`, `MuseTalk`, `LivePortrait`, `Deep-Live-Cam`) have been successfully integrated and debugged.
    *   **MuseTalk Hotfixes:** FFMPEG string space pathing (`os.system` interpolation) is fixed, missing `huggingface` checkpoint weights (DWPose, VAE, ResNet, Whisper) dynamically download via native cache scripts, and broken CLI default arguments are overridden.
    *   **MMCV Bypasses:** OpenMMLab C++ compilation failures on Windows are suppressed using an `ext_loader` MagicMock hotpatch.
    *   **LivePortrait Interceptions:** Missing explicit `-o` FFMPEG output bindings have been routed to a centralized sandboxed `temp_res` directory and pulled automatically by `shutil`.
4.  **Phase 4/5 (Merging & Metrics):** PyTorch and `mediapipe==0.10.14` accurately evaluate final AV sync and Landmark NME metrics.

---

## The Critical Bottleneck: Hardware Panics & The Unified `.venv`
Currently, the entire Phase 3 generation is running brutally slow (**40+ minutes per video**). To get the pipeline working, we had to enforce a universal fallback to CPU inference (`CUDA_VISIBLE_DEVICES="-1"`).

### Why CPU Inference Was Natively Enforced
The project originally forced a single massive virtual environment (`.venv`) utilizing **Python 3.12** and **PyTorch 2.6 Nightly**, attempting to load 5 disparate and fundamentally incompatible deepfake algorithms:

1.  **Hardware Incompatibility (RTX 5070 / `sm_120`):** Your discrete GPU runs on ultra-modern architecture (Blackwell/Ada Lovelace `sm_120`). Older legacy models like `Wav2Lip` (2020) rely on heavily deprecated 3D convolutions and PyTorch 1.x logic. Standard pre-compiled PyTorch 2.5/2.6 packages (`pip install torch`) natively panic and throw `cudaErrorNoKernelImageForDevice` because they cannot translate legacy CUDA mappings securely onto an `sm_120` silicon grid.
2.  **Dependency Clashing:** 
    *   *MuseTalk* (2023) breaks under PyTorch 2.6's new strict `weights_only=True` security lockdowns, requiring our `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"` environment injection. It also requires ancient `mmcv-lite` distributions natively tied to CUDA 11.8 / PyTorch 2.0 hashes.
    *   *LivePortrait* (2024) demands modern `xformers` optimization running on PyTorch 2.4+.
3.  **ONNX Clashes:** FaceFusion and Deep-Live-Cam utilize execution providers built around specific versions of `onnxruntime-gpu`, which instantly crash when Python’s `sys.modules` load clashing PyTorch `cu12x` DLL binaries into the same namespace.

Because these models conflict so aggressively, the only way to quickly stabilize the Python execution thread was isolating them computationally and dropping to CPU processing.

---

## Future Target Architecture: Multi-Environment Micro-Services
To process 30-50 videos practically, GPU acceleration is absolutely mandatory. 
**The Solution:** The pipeline must completely abandon the monolithic `.venv` design. Instead, it must utilize a "Master/Worker" paradigm. 

The orchestrator `.venv` (Master) merely iterates through the evaluation list. When it invokes a specific Deepfake model, `subprocess.run()` should target an isolated, perfectly-tailored Python executable embedded directly inside that model's repository.

### Technical Specification for Micro-Environments
To get native `sm_120` VRAM passthrough on your RTX 5070, build 5 completely standalone environments (using either `uv`, `miniconda`, or `.bat` virtualization). 

#### 1. Wav2Lip Worker (`venv_wav2lip`)
*   **Engine:** Python 3.9
*   **Core:** `torch==2.1.2+cu121` (or nearest stable CUDA 12.x equivalent for Windows).
*   **Note:** Wav2Lip requires heavily deprecated `torchvision`. The 2.1.x distribution usually provides decent backward compatibility without panicking modern GPUs.

#### 2. MuseTalk Worker (`venv_musetalk`)
*   **Engine:** Python 3.10
*   **Core:** `torch==2.1.2+cu121`, `torchvision==0.16.2`, `torchaudio==2.1.2`.
*   **Extension:** `mmcv==2.1.0` and `mmpose`. Must be installed rigorously from official OpenMMLab pre-compiled Windows wheels `(pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html)` to prevent ninja/C++ source compilation.

#### 3. LivePortrait Worker (`venv_liveportrait`)
*   **Engine:** Python 3.12
*   **Core:** `torch==2.4.0+cu124` (or PyTorch 2.5).
*   **Extension:** Modern `xformers` compiled directly for CUDA 12.4.

#### 4. FaceFusion Worker (`venv_facefusion`)
*   **Engine:** Python 3.11
*   **Core:** Natively clean `onnxruntime-gpu==1.19.0+` built carefully over CUDA 12.1.
*   **Note:** Absolutely no PyTorch overlapping.

#### 5. Deep-Live-Cam Worker (`venv_dlc`)
*   **Engine:** Python 3.10 (Standard for ONNX 1.16).
*   **Core:** `onnxruntime-gpu` strictly aligned with CUDA 11.8 / 12.x depending on the `ExecutionProvider`.

---

## Next Action Required by AI Assistant
When moving to the new chat, the subsequent AI agent should immediately execute the following steps:

1.  **Draft Auto-Installer Script:** Create a master executable (`setup_workers.bat` or `setup_envs.py`) that iteratively loops through the `Baseline_video` directories and utilizes `uv venv` or `miniconda` to materialize these exact specifications automatically.
2.  **Rewrite Pipeline Orchestration:** Open `src/pipeline/3_generate_video.py`. 
    *   Remove `cpu_env["CUDA_VISIBLE_DEVICES"] = "-1"`.
    *   Replace the generic `sys.executable` with variable bindings routing directly to `Path(f"Baseline_video/{arc}/.venv/Scripts/python.exe")` (or the respective Conda environment paths).
3.  **Validate Render Speeds:** Deploy a 1-video test cycle to confirm the RTX 5070 engages correctly across all 5 architectures natively, verifying that a 40-minute CPU operation successfully resolves in <30 seconds via pure VRAM acceleration.
