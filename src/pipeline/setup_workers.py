import subprocess
import shutil
import re
from pathlib import Path

# Explicitly define the compatible Python version for each repo
WORKERS = {
    "Applio": {"repo": "https://github.com/IAHispano/Applio.git", "dir": "audio/Applio", "req": "requirements.txt", "py": "3.10"},
    "FaceFusion": {"repo": "https://github.com/facefusion/facefusion.git", "dir": "Baseline_video/facefusion", "req": "requirements.txt", "py": "3.11"},
    "Wav2Lip": {"repo": "https://github.com/Rudrabha/Wav2Lip.git", "dir": "Baseline_video/Wav2Lip", "req": "requirements.txt", "py": "3.10"},
    "MuseTalk": {"repo": "https://github.com/TMElyralab/MuseTalk.git", "dir": "Baseline_video/MuseTalk", "req": "requirements.txt", "py": "3.10"},
    "LivePortrait": {"repo": "https://github.com/KwaiVGI/LivePortrait.git", "dir": "Baseline_video/LivePortrait", "req": "requirements.txt", "py": "3.10"},
    "Deep-Live-Cam": {"repo": "https://github.com/hacksider/Deep-Live-Cam.git", "dir": "realtime_video/Deep-Live-Cam", "req": "requirements.txt", "py": "3.10"}
}

def run_cmd(cmd, cwd=None):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def sanitize_requirements(req_path):
    """ONLY removes PyTorch. We leave everything else intact because we are matching their native Python version."""
    with open(req_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(req_path, 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Drop torch so we can inject our Ada 6000 compatible cu124 versions
            if re.match(r'^(torch|torchvision|torchaudio)\b', line, re.IGNORECASE):
                continue
                
            f.write(line + '\n')

def main():
    print("="*70 + "\n  Initializing Linux Sub-Repositories and Virtual Environments\n" + "="*70)
    for name, info in WORKERS.items():
        target_dir = Path(info["dir"])
        req_path = target_dir / info["req"]
        py_version = info["py"]
        
        # 1. Clone or Re-clone if broken/empty
        if not target_dir.exists() or not (target_dir / ".git").exists():
            print(f"\n[!] Missing or broken repository for {name}. Cloning...")
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            run_cmd(["git", "clone", info["repo"], str(target_dir)])
        else:
            print(f"\n[*] {name} repository found.")
        
        # 2. Setup Venv with SPECIFIC Python Version
        venv_dir = target_dir / ".venv"
        venv_python = venv_dir / "bin" / "python"
        
        if not venv_dir.exists():
            print(f"[+] Creating Python {py_version} venv for {name}...")
            # uv will dynamically download the requested Python runtime if it doesn't exist
            run_cmd(["uv", "venv", "--python", py_version, ".venv"], cwd=target_dir)
        
        # 3. Install Requirements
        if req_path.exists():
            print(f"[+] Sanitizing PyTorch requirements for {name}...")
            sanitize_requirements(req_path)

            print(f"[+] Installing requirements for {name}...")
            run_cmd([
                "uv", "pip", "install", 
                "--python", str(venv_python.absolute()),
                # Inject stable CUDA 12.4 PyTorch for the Ada 6000
                "torch==2.5.1+cu124", "torchvision==0.20.1+cu124", "torchaudio==2.5.1+cu124",
                "-r", info["req"],
                "--index-strategy", "unsafe-best-match",
                "--extra-index-url", "https://download.pytorch.org/whl/cu124"
            ], cwd=target_dir)
            
            # Explicitly install ONNX Runtime GPU for architectures that need it
            if name in ["FaceFusion", "Deep-Live-Cam"]:
                print(f"[+] Injecting ONNX Runtime GPU for {name}...")
                run_cmd([
                    "uv", "pip", "install", 
                    "--python", str(venv_python.absolute()),
                    "onnxruntime-gpu"
                ], cwd=target_dir)

    print("\n" + "="*70)
    print("  Setup Complete! You can now run the pipeline.")
    print("="*70)

if __name__ == "__main__":
    main()