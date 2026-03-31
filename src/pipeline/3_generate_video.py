import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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

    manifest_path = Path(args.manifest)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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