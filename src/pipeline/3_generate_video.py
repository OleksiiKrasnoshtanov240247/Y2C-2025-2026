import argparse
import json
import os
<<<<<<< HEAD
=======
import shutil
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
import subprocess
import sys
from pathlib import Path

<<<<<<< HEAD
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
    
=======
import yaml

from utils import build_worker_env, get_best_gpu

WORKER_DIRS: dict[str, str] = {
    "FaceFusion":    "Baseline_video/facefusion",
    "Wav2Lip":       "Baseline_video/Wav2Lip",
    "MuseTalk":      "Baseline_video/MuseTalk",
    "LivePortrait":  "Baseline_video/LivePortrait",
    "Deep-Live-Cam": "realtime_video/Deep-Live-Cam",
}

ARCHITECTURES = list(WORKER_DIRS.keys())

TARGET_FACE = Path("src/assets/target_image/elunma.jpeg")


def get_worker_python(arc: str) -> str:
    return str(Path(WORKER_DIRS[arc]).absolute() / ".venv/bin/python")


# ---------------------------------------------------------------------------
# Per-architecture runners
# ---------------------------------------------------------------------------

def run_facefusion(
    vid_id: str, source_video: Path, audio_path: Path, out_path: Path, env: dict
) -> bool:
    """
    FaceFusion -o expects a full file path whose parent directory already exists.
    The error "specify the output image or video within a directory" means the
    parent dir doesn't exist yet — FaceFusion validates this before running.
    We mkdir the parent explicitly before launching.
    """
    # Parent must exist before FaceFusion validates the output path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        get_worker_python("FaceFusion"), "facefusion.py", "headless-run",
        "-s", str(TARGET_FACE.absolute()),
        "-t", str(source_video),
        "-o", str(out_path.absolute()),   # full file path, parent must pre-exist
        "--processors", "face_swapper", "face_enhancer",
        "--execution-providers", "cuda",
        "--execution-thread-count", "1",
    ]
    result = subprocess.run(cmd, env=env, cwd=WORKER_DIRS["FaceFusion"], check=False)
    if result.returncode != 0:
        print(f"  [ERROR] FaceFusion exited with code {result.returncode}")
        return False

    if out_path.exists() and out_path.stat().st_size > 0:
        return True

    print(f"  [ERROR] FaceFusion produced no output at {out_path.name}")
    return False


def run_wav2lip(
    vid_id: str, source_video: Path, audio_path: Path, out_path: Path, env: dict
) -> bool:
    """
    Wav2Lip's inference.py calls ffmpeg internally with shell=True from its own
    cwd.  It converts the outfile path to relative internally (os.path.relpath),
    which breaks if the output directory doesn't exist relative to Wav2Lip's cwd.

    Fix: route output through Wav2Lip's own results/ directory using an absolute
    path that IS local to Wav2Lip's cwd, then move to our output location.
    """
    worker_abs = Path(WORKER_DIRS["Wav2Lip"]).absolute()
    Path(WORKER_DIRS["Wav2Lip"] + "/temp").mkdir(parents=True, exist_ok=True)

    # Write into Wav2Lip's own results/ so the path is always valid from its cwd
    wav2lip_results = worker_abs / "results"
    wav2lip_results.mkdir(parents=True, exist_ok=True)
    wav2lip_out = wav2lip_results / f"_out_{out_path.stem}.mp4"

    cmd = [
        get_worker_python("Wav2Lip"), "inference.py",
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
        "--face", str(source_video),
        "--audio", str(audio_path),
        "--outfile", str(wav2lip_out),  # absolute path inside Wav2Lip dir
    ]
    result = subprocess.run(cmd, env=env, cwd=WORKER_DIRS["Wav2Lip"], check=False)
    if result.returncode != 0:
        print(f"  [ERROR] Wav2Lip exited with code {result.returncode}")
        return False
    if not wav2lip_out.exists() or wav2lip_out.stat().st_size == 0:
        print(f"  [ERROR] Wav2Lip produced no output at {wav2lip_out.name}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(wav2lip_out), str(out_path))
    return True


def run_musetalk(
    vid_id: str, source_video: Path, audio_path: Path, out_path: Path, env: dict
) -> bool:
    """
    MuseTalk crashes on paths with spaces or parentheses in its internal ffmpeg
    call.  We stage a clean-named copy in the worker dir, run inference, then
    move the newest result mp4 to our output location.
    """
    worker_dir = Path(WORKER_DIRS["MuseTalk"])
    safe_video = worker_dir / f"_stage_{vid_id}.mp4"
    shutil.copy(str(source_video), safe_video)

    yaml_config = worker_dir / "configs/inference/temp_pipeline.yaml"
    yaml_config.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_config, "w") as f:
        yaml.dump(
            {"task_0": {
                "video_path": str(safe_video.absolute()),
                "audio_path": str(audio_path),
                "bbox_shift": 0,
            }},
            f,
        )

    cmd = [
        get_worker_python("MuseTalk"), "-m", "scripts.inference",
        "--inference_config", "configs/inference/temp_pipeline.yaml",
        "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json",
    ]

    try:
        result = subprocess.run(cmd, cwd=str(worker_dir), env=env, check=False)
    finally:
        if safe_video.exists():
            safe_video.unlink()

    if result.returncode != 0:
        print(f"  [ERROR] MuseTalk exited with code {result.returncode}")
        return False

    results_dir = worker_dir / "results"
    candidates = list(results_dir.rglob("*.mp4")) if results_dir.exists() else []
    if not candidates:
        print(f"  [ERROR] MuseTalk produced no mp4 in {results_dir}")
        return False

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    shutil.move(str(candidates[0]), str(out_path))
    return True


def run_liveportrait(
    vid_id: str, source_video: Path, audio_path: Path, out_path: Path, env: dict
) -> bool:
    worker_dir = Path(WORKER_DIRS["LivePortrait"])
    temp_out = worker_dir / f"_temp_res_{vid_id}"
    temp_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        get_worker_python("LivePortrait"), "inference.py",
        "-s", str(TARGET_FACE.absolute()),
        "-d", str(source_video),
        "-o", str(temp_out.absolute()),
    ]

    try:
        result = subprocess.run(cmd, env=env, cwd=str(worker_dir), check=False)

        if result.returncode != 0:
            print(f"  [ERROR] LivePortrait exited with code {result.returncode}")
            return False

        # LivePortrait produces several variants; we want the plain animated video
        # (not _concat or _concat_with_audio which include the source side-by-side).
        candidates = [
            p for p in temp_out.rglob("*.mp4")
            if "_concat" not in p.name and "_with_audio" not in p.name
        ]
        if not candidates:
            # Fallback: any mp4
            candidates = list(temp_out.rglob("*.mp4"))
        if not candidates:
            print(f"  [ERROR] LivePortrait produced no mp4 in {temp_out}")
            return False

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        shutil.move(str(candidates[0]), str(out_path))
        return True
    finally:
        if temp_out.exists():
            shutil.rmtree(temp_out, ignore_errors=True)


def run_deep_live_cam(
    vid_id: str, source_video: Path, audio_path: Path, out_path: Path, env: dict
) -> bool:
    """
    Deep-Live-Cam has two path-related issues:

    1. Output path must be absolute.  DLC's shutil.move uses the output path as
       a destination; if it's relative and the cwd doesn't contain the parent
       directory, it fails with FileNotFoundError.

    2. Temp file location.  DLC creates temp frames at
       {source_video_dir}/temp/{source_stem}/ and assembles temp.mp4 there.
       After assembly it calls restore_audio which moves temp.mp4 to the output
       path.  If the source video directory is read-only or if DLC resolves paths
       differently, temp.mp4 may end up inside DLC's own cwd instead.

    We pass an absolute output path and add a fallback search for temp.mp4 in
    case DLC's internal move fails (which it will report as a non-zero exit, but
    the video may still have been encoded correctly).
    """
    worker_abs = Path(WORKER_DIRS["Deep-Live-Cam"]).absolute()
    out_abs = out_path.absolute()
    out_abs.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        get_worker_python("Deep-Live-Cam"), "run.py",
        "-s", str(TARGET_FACE.absolute()),
        "-t", str(source_video),
        "-o", str(out_abs),  # absolute — DLC's shutil.move needs this
        "--execution-provider", "cuda",
    ]
    result = subprocess.run(cmd, cwd=str(worker_abs), env=env, check=False)

    # Check primary output first (DLC succeeded cleanly)
    if out_abs.exists() and out_abs.stat().st_size > 0:
        return True

    if result.returncode != 0:
        print(f"  [ERROR] Deep-Live-Cam exited with code {result.returncode}")

    # Fallback: DLC encoded the video but move_temp failed (common when source
    # video is in a read-only directory).  Search known temp locations.
    # Priority: next to source video, then inside DLC's own cwd.
    temp_candidates: list[Path] = []
    for search_root in [
        source_video.parent / "temp" / source_video.stem,
        worker_abs / "temp" / source_video.stem,
        worker_abs / "temp",
    ]:
        if search_root.exists():
            temp_candidates.extend(search_root.rglob("*.mp4"))

    if temp_candidates:
        temp_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"  [WARN] DLC move failed — recovering from {temp_candidates[0].name}")
        shutil.move(str(temp_candidates[0]), str(out_abs))
        return True

    print(f"  [ERROR] Deep-Live-Cam produced no recoverable output")
    return False


RUNNERS = {
    "FaceFusion":    run_facefusion,
    "Wav2Lip":       run_wav2lip,
    "MuseTalk":      run_musetalk,
    "LivePortrait":  run_liveportrait,
    "Deep-Live-Cam": run_deep_live_cam,
}


def main():
    best_gpu = get_best_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = best_gpu

    parser = argparse.ArgumentParser(description="Phase 3: Video Generation")
    parser.add_argument("--manifest", type=str, default="data/dataset_manifest.json")
    parser.add_argument("--audio_dir", type=str, default="data/generated_audio")
    parser.add_argument("--output_dir", type=str, default="data/generated_video")
    args = parser.parse_args()

>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
    manifest_path = Path(args.manifest)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
<<<<<<< HEAD
    
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
=======

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}. Run Phase 1 first.")
        sys.exit(1)

    if not TARGET_FACE.exists():
        print(f"[ERROR] Target face image not found: {TARGET_FACE}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    videos = manifest.get("videos", {})
    if not videos:
        print("[WARN] No videos in manifest.")
        sys.exit(0)

    # Build worker envs once — not inside the loop
    worker_envs: dict[str, dict] = {}
    for arc in ARCHITECTURES:
        env = build_worker_env(get_worker_python(arc))
        env["CUDA_VISIBLE_DEVICES"] = best_gpu
        worker_envs[arc] = env

    print("\n--- Starting Video Generation ---")

    for vid_id, data in videos.items():
        source_video = data.get("source_video")
        if not source_video or not Path(source_video).exists():
            print(f"[WARN] Source video missing for {vid_id}, skipping.")
            continue

        source_video_path = Path(source_video).absolute()
        matched_audios = sorted(audio_dir.glob(f"{vid_id}_*.wav"))

        if not matched_audios:
            print(f"[WARN] No generated audio for {vid_id}, skipping.")
            continue

        for audio_path in matched_audios:
            condition_suffix = audio_path.stem[len(vid_id) + 1:]

            for arc in ARCHITECTURES:
                out_path = output_dir / f"{vid_id}_{condition_suffix}_{arc}.mp4"

                if out_path.exists() and out_path.stat().st_size > 0:
                    print(f"  [SKIP] {out_path.name}")
                    continue

                print(f"\n  [{arc}] {vid_id} / {condition_suffix}")

                ok = RUNNERS[arc](
                    vid_id,
                    source_video_path,
                    audio_path.absolute(),
                    out_path,
                    worker_envs[arc],
                )

                if ok:
                    print(f"  [OK]   {out_path.name}")
                else:
                    print(f"  [FAIL] {arc} for {vid_id}/{condition_suffix}")


if __name__ == "__main__":
    main()
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
