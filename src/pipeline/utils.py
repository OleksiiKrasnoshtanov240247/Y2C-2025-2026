"""
Shared utilities for the deepfake evaluation pipeline.
Imported by individual phase scripts — kept here to avoid duplication
across phases 2 and 3 which both need GPU selection and worker env building.
"""
import glob
import os
import subprocess
from pathlib import Path


def get_best_gpu() -> str:
    """
    Query nvidia-smi and return the index (as str) of the GPU with the most
    free VRAM.  Falls back to "0" on any error so the pipeline still runs
    on single-GPU machines or when smi is unavailable.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        entries = []
        for line in output.strip().split("\n"):
            idx, free = line.split(", ")
            entries.append((int(idx), int(free)))
        entries.sort(key=lambda x: x[1], reverse=True)
        best = str(entries[0][0])
        print(f"[GPU] Auto-selected GPU {best} ({entries[0][1]} MB free)")
        return best
    except Exception as exc:
        print(f"[GPU] nvidia-smi failed, defaulting to GPU 0. Reason: {exc}")
        return "0"


def build_worker_env(venv_python: str) -> dict:
    """
    Build a subprocess environment dict for a worker venv.

    Why this is needed:
      Worker venvs install CUDA libraries (cublas, cudnn) via pip into their
      own site-packages/nvidia/ tree.  Without prepending these to
      LD_LIBRARY_PATH, ONNX Runtime and PyTorch can't resolve the shared libs
      at runtime, even though they're physically present on disk.

    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 suppresses the torch.load safety
    warning that some older checkpoint files trigger on PyTorch >= 2.0.
    """
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    # Derive site-packages from the venv python binary path
    venv_base = venv_python.replace("/bin/python", "")
    sp_matches = glob.glob(f"{venv_base}/lib/python3.*/site-packages")

    if sp_matches:
        sp = sp_matches[0]
        lib_dirs = [
            f"{sp}/nvidia/cublas/lib",
            f"{sp}/nvidia/cudnn/lib",
            f"{sp}/torch/lib",
        ]
        prepend = [d for d in lib_dirs if Path(d).exists()]
        existing = env.get("LD_LIBRARY_PATH", "")
        parts = prepend + ([existing] if existing else [])
        env["LD_LIBRARY_PATH"] = ":".join(parts)

    return env
