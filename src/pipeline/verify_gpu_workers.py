import sys
import subprocess
from pathlib import Path

WORKERS = [
    {"name": "Wav2Lip", "dir": "Baseline_video/Wav2Lip"},
    {"name": "MuseTalk", "dir": "Baseline_video/MuseTalk"},
    {"name": "LivePortrait", "dir": "Baseline_video/LivePortrait"},
    {"name": "facefusion", "dir": "Baseline_video/facefusion"},
    {"name": "Deep-Live-Cam", "dir": "realtime_video/Deep-Live-Cam"}
]

print("="*60)
print("  Verifying Micro-Environment GPU Compatibilities")
print("="*60)

for worker in WORKERS:
    # LINUX: .venv/bin/python
    venv_python = Path(worker["dir"]).absolute() / ".venv" / "bin" / "python"
    
    if not venv_python.exists():
        print(f"[ERROR] Engine missing: {worker['name']} ({venv_python} not found)")
        continue
        
    print(f"\nProbing {worker['name']}...")
    
    if worker["name"] in ["facefusion", "Deep-Live-Cam"]:
        probe_code = "import onnxruntime; print('ONNX Providers:', onnxruntime.get_available_providers())"
        cmd = [str(venv_python), "-c", probe_code]
    else:
        probe_code = "import torch; print(f'CUDA Connected: {torch.cuda.is_available()} | Version: {torch.version.cuda} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
        cmd = [str(venv_python), "-c", probe_code]
        
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  [OK] {res.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] Subprocess Error: {e.stderr.strip()}")
    except Exception as e:
        print(f"  [FAIL] System Error: {e}")

print("\nVerification sequence finished.")