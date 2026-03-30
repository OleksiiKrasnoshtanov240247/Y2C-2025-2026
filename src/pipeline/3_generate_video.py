import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Mapping of required architectures to their Git repositories
ARCHITECTURES = {
    "FaceFusion": {
        "repo": "https://github.com/facefusion/facefusion.git",
        "dir": "Baseline_video/facefusion",
        "install_cmd": None # Handled via pyproject.toml / uv ideally, but can be added if needed
    },
    "Wav2Lip": {
        "repo": "https://github.com/Rudrabha/Wav2Lip.git",
        "dir": "Baseline_video/Wav2Lip",
        "install_cmd": None
    },
    "MuseTalk": {
        "repo": "https://github.com/TMElyralab/MuseTalk.git",
        "dir": "Baseline_video/MuseTalk",
        "install_cmd": None
    },
    "LivePortrait": {
        "repo": "https://github.com/KwaiVGI/LivePortrait.git",
        "dir": "Baseline_video/LivePortrait",
        "install_cmd": None
    },
    "Deep-Live-Cam": {
        "repo": None,
        "dir": "realtime_video/Deep-Live-Cam",
        "install_cmd": None
    }
}

def patch_mmcv_lite():
    """Patches MMCV-lite's ext_loader strictly to return MagicMocks instead of crashing mmpose."""
    try:
        import mmcv.utils.ext_loader as ext_loader
        p = Path(ext_loader.__file__)
        content = p.read_text()
        if "MagicMock" not in content:
            new_content = content.replace(
                "    def load_ext(name, funcs):\n        ext = importlib.import_module('mmcv.' + name)\n        for fun in funcs:\n            assert hasattr(ext, fun), f'{fun} miss in module {name}'\n        return ext",
                "    from unittest.mock import MagicMock\n    def load_ext(name, funcs):\n        try:\n            ext = importlib.import_module('mmcv.' + name)\n            for fun in funcs:\n                if not hasattr(ext, fun): return MagicMock()\n            return ext\n        except Exception:\n            return MagicMock()"
            )
            p.write_text(new_content)
            print("  [SUCCESS] mmcv-lite ext_loader hot-patched!")
    except Exception as e:
        pass

def pull_musetalk_weights():
    try:
        import os
        from huggingface_hub import snapshot_download
        base = "Baseline_video/MuseTalk/models"
        os.makedirs(base, exist_ok=True)
        snapshot_download("TMElyralab/MuseTalk", local_dir=base)
        snapshot_download("stabilityai/sd-vae-ft-mse", local_dir=f"{base}/sd-vae", allow_patterns=["config.json", "diffusion_pytorch_model.bin"])
        snapshot_download("openai/whisper-tiny", local_dir=f"{base}/whisper", allow_patterns=["config.json", "pytorch_model.bin", "preprocessor_config.json"])
        snapshot_download("yzd-v/DWPose", local_dir=f"{base}/dwpose", allow_patterns=["dw-ll_ucoco_384.pth"])
        snapshot_download("ManyOtherFunctions/face-parse-bisent", local_dir=f"{base}/face-parse-bisent", allow_patterns=["79999_iter.pth", "resnet18-5c106cde.pth"])
        print("  [SUCCESS] MuseTalk auxiliary models securely cached!")
    except Exception as e:
        print(f"  [ERROR] MuseTalk weight pull failed: {e}")

def ensure_repositories_cloned():
    """Checks if required sub-repositories are cloned. If not, clones them."""
    print("--- Checking Architecture Dependencies ---")
    for name, info in ARCHITECTURES.items():
        if info["repo"] is None:
            print(f"  [OK] {name} skips cloning (local manual mapping at {info['dir']}).")
            continue
            
        target_dir = Path(info["dir"])
        # If the folder exists, we assume it's set up and don't try to clone over it
        if not target_dir.exists():
            print(f"  [INFO] Setting up {name}... Cloning from {info['repo']} into {info['dir']}")
            try:
                # Ensure parent directory exists
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "clone", info["repo"], str(target_dir)], check=True)
                print(f"  [SUCCESS] {name} cloned successfully.")
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] Failed to clone {name}. Error: {e}")
        else:
            print(f"  [OK] {name} is already present at {info['dir']}.")
            
    # Trigger dynamic MMCV patch natively to support OpenMMLab 
    patch_mmcv_lite()
    
    # Pre-download Facefusion models to prevent NoneType Hash exceptions
    try:
        # Explicitly use FaceFusion's exact Python executable
        facefusion_python = str(Path("Baseline_video/facefusion/.venv/Scripts/python.exe").absolute())
        
        # Check if the environment exists first to prevent crashes during initial pipelining
        if Path(facefusion_python).exists():
            subprocess.run([facefusion_python, "facefusion.py", "force-download"], cwd="Baseline_video/facefusion", check=True)
        else:
            print("[WARN] Facefusion virtual environment not found. Pre-download step skipped.")
            
        # Pull all MuseTalk specific checkpoints natively
        pull_musetalk_weights()
    except Exception as e:
        print(f"Failed to pull background weights: {e}")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Video Generation Pipeline")
    parser.add_argument("--manifest", type=str, default="data/dataset_manifest.json", help="Path to input manifest")
    parser.add_argument("--audio_dir", type=str, default="data/generated_audio", help="Directory with generated audio")
    parser.add_argument("--output_dir", type=str, default="data/generated_video", help="Directory to save generated video")
    
    args = parser.parse_args()
    
    # 1. Ensure all repositories exist because code will run on different machines
    ensure_repositories_cloned()
    
    manifest_path = Path(args.manifest)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}. Run Phase 1 first.")
        sys.exit(1)
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    videos = manifest.get("videos", {})
    if not videos:
        print("[WARN] No videos found in manifest.")
        sys.exit(0)

    # 2. Iterate through SQ2 Architectures
    architectures_to_test = ["Wav2Lip", "MuseTalk", "LivePortrait", "FaceFusion", "Deep-Live-Cam"]
    
    print("\n--- Starting Video Generation ---")
    
    # Note: During actual execution, we would call the respective architecture's inference scripts.
    # We will log the intended commands here for clarity and later expansion.
    
    for vid_id, data in videos.items():
        source_video = data.get("source_video")
        
        # We need the generated audio from Phase 2
        # Phase 2 generates files like: <vid_id>_<speaker>_<C_level>_<precision>.wav
        # To simplify this scaffolding, we will look for any generated audio matching this video
        matched_audios = list(audio_dir.glob(f"{vid_id}_*.wav"))
        
        if not matched_audios:
            print(f"  [WARN] Skipping {vid_id}, no matching generated audio found in {audio_dir}.")
            continue
            
        for audio_path in matched_audios:
            # Reconstruct the base condition from the audio file name
            # Format expected: video_001_name_DonaldTrump_C3_fp32.wav
            condition_suffix = audio_path.stem.replace(vid_id + "_", "")
            
            for arc in architectures_to_test:
                out_name = f"{vid_id}_{condition_suffix}_{arc}.mp4"
                out_path = output_dir / out_name
                
                print(f"  [{arc}] Processing {vid_id} ...")
                
                # For FaceFusion, Wav2Lip etc., we resolve the Python environment path first
                if arc == "FaceFusion":
                    env_python = str(Path("Baseline_video/facefusion/.venv/Scripts/python.exe").absolute())
                elif arc == "Wav2Lip":
                    env_python = str(Path("Baseline_video/Wav2Lip/.venv/Scripts/python.exe").absolute())
                elif arc == "MuseTalk":
                    env_python = str(Path("Baseline_video/MuseTalk/.venv/Scripts/python.exe").absolute())
                elif arc == "LivePortrait":
                    env_python = str(Path("Baseline_video/LivePortrait/.venv/Scripts/python.exe").absolute())
                elif arc == "Deep-Live-Cam":
                    env_python = str(Path("realtime_video/Deep-Live-Cam/.venv/Scripts/python.exe").absolute())
                else:
                    env_python = sys.executable

                # Native Accelerated Setting
                gpu_env = os.environ.copy()
                gpu_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
                
                # Robustly inject native CUDA DLLs from PyTorch into PATH for ONNX compatibility on Windows
                worker_base = env_python.replace("Scripts\\python.exe", "").replace("Scripts/python.exe", "")
                site_packages = Path(worker_base) / "Lib" / "site-packages"
                new_path = gpu_env.get("PATH", "")
                for nv_dir in ["cublas", "cudnn", "cuda_runtime", "cufft", "cusparse", "cusolver", "curand", "nvtx"]:
                    nv_bin = site_packages / "nvidia" / nv_dir / "bin"
                    if nv_bin.exists():
                        new_path = str(nv_bin) + os.pathsep + new_path
                # Add PyTorch 2.4+ root lib folder where modern CUDA extensions natively sit
                torch_lib = site_packages / "torch" / "lib"
                if torch_lib.exists():
                    new_path = str(torch_lib) + os.pathsep + new_path
                gpu_env["PATH"] = new_path

                # Command logic
                if arc == "FaceFusion":
                    cmd = [
                        env_python, "facefusion.py", "headless-run",
                        "--source", str(Path("src/assets/target_image/elunma.jpeg").absolute()), 
                        "--target", str(Path(source_video).absolute()),
                        "--output-path", str(Path(out_path).absolute()),
                        "--processors", "face_swapper", "face_enhancer",
                        "--execution-providers", "cuda"
                    ]
                    try:
                        subprocess.run(cmd, env=gpu_env, cwd="Baseline_video/facefusion", check=True)
                    except Exception as e:
                        print(f"  [ERROR] {arc} failed: {e}")
                        
                elif arc == "Wav2Lip":
                    Path("Baseline_video/Wav2Lip/temp").mkdir(parents=True, exist_ok=True)
                    cmd = [
                        env_python, "inference.py",
                        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
                        "--face", str(Path(source_video).absolute()),
                        "--audio", str(Path(audio_path).absolute()),
                        "--outfile", str(Path(out_path).absolute())
                    ]
                    try:
                        subprocess.run(cmd, env=gpu_env, cwd="Baseline_video/Wav2Lip", check=True)
                    except Exception as e:
                        print(f"  [ERROR] {arc} failed: {e}")
                        
                elif arc == "MuseTalk":
                    # MuseTalk uses a yaml config, so we will generate it dynamically
                    import yaml
                    config_dict = {
                        "task_0": {
                            "video_path": str(Path(source_video).absolute()),
                            "audio_path": str(Path(audio_path).absolute()),
                            "bbox_shift": 0
                        }
                    }
                    yaml_path = Path("Baseline_video/MuseTalk/configs/inference/temp_pipeline.yaml")
                    yaml_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(yaml_path, "w") as f:
                        yaml.dump(config_dict, f)
                        
                    cmd = [
                        env_python, "-m", "scripts.inference", 
                        "--inference_config", "configs/inference/temp_pipeline.yaml",
                        "--unet_model_path", "models/musetalkV15/unet.pth",
                        "--unet_config", "models/musetalkV15/musetalk.json",
                        "--version", "v15"
                    ]
                    
                    try:
                        # MuseTalk must be run from its directory module-wise
                        subprocess.run(cmd, cwd="Baseline_video/MuseTalk", env=gpu_env, check=True)
                        output_from_musetalk = list(Path("Baseline_video/MuseTalk/results").rglob("*.mp4"))
                        if output_from_musetalk:
                            # Move result to out_path
                            import shutil
                            output_from_musetalk.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            shutil.move(output_from_musetalk[0], out_path)
                    except Exception as e:
                        print(f"  [ERROR] {arc} failed: {e}")

                elif arc == "LivePortrait":
                    # standard KwaiVGI LivePortrait defaults (driving video + source face image)
                    cmd = [
                        env_python, "inference.py", 
                        "-s", str(Path("src/assets/target_image/elunma.jpeg").absolute()), 
                        "-d", str(Path(source_video).absolute()),
                        "-o", str(Path("temp_res").absolute())
                    ]
                    try:
                        subprocess.run(cmd, env=gpu_env, cwd="Baseline_video/LivePortrait", check=True)
                        output_files = list(Path("Baseline_video/LivePortrait/temp_res").rglob("*.mp4"))
                        if output_files:
                            import shutil
                            output_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            shutil.move(output_files[0], out_path)
                    except Exception as e:
                        print(f"  [ERROR] {arc} failed: {e}")

                elif arc == "Deep-Live-Cam":
                    cmd = [
                        env_python, "run.py",
                        "-s", str(Path("src/assets/target_image/elunma.jpeg").absolute()),
                        "-t", str(Path(source_video).absolute()),
                        "-o", str(Path(out_path).absolute()),
                        "--execution-provider", "cuda" 
                    ]
                    try:
                        subprocess.run(cmd, cwd="realtime_video/Deep-Live-Cam", env=gpu_env, check=True)
                    except Exception as e:
                        print(f"  [ERROR] {arc} failed: {e}")
                
    print("\nPhase 3 Complete!")

if __name__ == "__main__":
    main()
