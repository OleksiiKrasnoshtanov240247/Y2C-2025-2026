import os
import subprocess
from pathlib import Path
import shutil
import re

WORKERS = [
    {
        "name": "Wav2Lip",
        "dir": "Baseline_video/Wav2Lip",
        "python": "3.9",
        "pre_install": [
            "setuptools", "torch==2.6.0+cu126", "torchvision==0.21.0+cu126", "torchaudio==2.6.0+cu126",
            "--extra-index-url", "https://download.pytorch.org/whl/cu126"
        ],
        "post_install": [
            ["opencv-python", "opencv-contrib-python", "numpy<2"],
            ["librosa==0.9.2"]
        ],
        "ignore_reqs": ["torch", "torchvision", "torchaudio", "opencv-python", "opencv-contrib-python", "numba", "numpy"]
    },
    {
        "name": "MuseTalk",
        "dir": "Baseline_video/MuseTalk",
        "python": "3.10",
        "pre_install": [
            "setuptools", "torch==2.6.0+cu126", "torchvision==0.21.0+cu126", "torchaudio==2.6.0+cu126",
            "--extra-index-url", "https://download.pytorch.org/whl/cu126"
        ],
        "post_install": [
            [{"MMCV_WITH_OPS": "0"}, "--no-build-isolation", "mmcv==2.1.0"],
            ["mmpose", "mmdet", "mmeval"]
        ],
        "ignore_reqs": ["torch", "torchvision", "torchaudio", "mmcv", "mmpose", "mmdet", "mmeval"]
    },
    {
        "name": "LivePortrait",
        "dir": "Baseline_video/LivePortrait",
        "python": "3.11",
        "pre_install": [
            "torch==2.6.0+cu126", "torchvision==0.21.0+cu126", "torchaudio==2.6.0+cu126",
            "--extra-index-url", "https://download.pytorch.org/whl/cu126"
        ],
        "post_install": [
            ["xformers==0.0.35"],
            ["lmdb"] # Unpinned lmdb because v1.4.1 lacks wheels for 3.12 and crashes PEP 517 build
        ],
        "ignore_reqs": ["torch", "torchvision", "torchaudio", "xformers", "lmdb"]
    },
    {
        "name": "FaceFusion",
        "dir": "Baseline_video/facefusion",
        "python": "3.11",
        "pre_install": [
            "onnxruntime-gpu==1.20.1", "torch==2.6.0+cu126", "torchvision==0.21.0+cu126", 
            "--extra-index-url", "https://download.pytorch.org/whl/cu126"
        ],
        "ignore_reqs": ["onnxruntime-gpu", "onnxruntime", "torch", "torchvision", "torchaudio"]
    },
    {
        "name": "Deep-Live-Cam",
        "dir": "realtime_video/Deep-Live-Cam",
        "python": "3.11",
        "pre_install": [
            "onnxruntime-gpu==1.20.1", "torch==2.6.0+cu126",
            "--extra-index-url", "https://download.pytorch.org/whl/cu126"
        ],
        "ignore_reqs": ["onnxruntime-gpu", "onnxruntime", "onnxruntime-silicon", "torch"]
    }
]

def clean_requirements(target_dir, req_file, ignore_list, clean_reqs_set):
    req_path = target_dir / req_file
    if not req_path.exists():
        return
        
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                sub_req = line.split("-r ")[1].strip()
                clean_requirements(target_dir, sub_req, ignore_list, clean_reqs_set)
                continue
                
            should_ignore = False
            for ig in ignore_list:
                if re.match(rf"^{ig}([^a-zA-Z0-9_\-]|$)", line, re.IGNORECASE):
                    should_ignore = True
                    break
            
            if not should_ignore:
                clean_reqs_set.add(line)

def run_cmd(cmd, env=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

def setup_worker(worker):
    print(f"\n{'='*70}")
    print(f"  Setting up {worker['name']} (Python {worker['python']})")
    print(f"{'='*70}")
    
    target_dir = Path(worker["dir"]).absolute()
    if not target_dir.exists():
        print(f"[WARN] Directory {target_dir} not found. Skipping...")
        return
        
    venv_dir = target_dir / ".venv"
    if venv_dir.exists():
        print(f"-> NUKING existing .venv in {target_dir} to enforce Python {worker['python']}...")
        shutil.rmtree(venv_dir)
        
    run_cmd(["uv", "venv", "--python", worker["python"], str(venv_dir)])
    
    python_exe = venv_dir / "Scripts" / "python.exe"
    pip_cmd = ["uv", "pip", "install", "--python", str(python_exe)]
    
    try:
        if "pre_install" in worker:
            run_cmd(pip_cmd + worker["pre_install"])
            
        clean_reqs_set = set()
        clean_requirements(target_dir, "requirements.txt", worker.get("ignore_reqs", []), clean_reqs_set)
        
        if clean_reqs_set:
            temp_req_path = target_dir / "temp_requirements_clean.txt"
            with open(temp_req_path, "w", encoding="utf-8") as f:
                for req in list(clean_reqs_set):
                    f.write(req + "\n")
            
            try:
                run_cmd(pip_cmd + ["-r", str(temp_req_path)])
            finally:
                if temp_req_path.exists():
                    temp_req_path.unlink()
                
        if "post_install" in worker:
            for item in worker["post_install"]:
                if isinstance(item[0], dict):
                    # We pass distinct environment variables for this compilation step
                    custom_env = os.environ.copy()
                    custom_env.update(item[0])
                    run_cmd(pip_cmd + item[1:], env=custom_env)
                else:
                    run_cmd(pip_cmd + item)
                
        print(f"[SUCCESS] {worker['name']} micro-environment cleanly finalized.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] Failed to setup {worker['name']} environment. See logs above.\n")

if __name__ == "__main__":
    for w in WORKERS:
        setup_worker(w)
