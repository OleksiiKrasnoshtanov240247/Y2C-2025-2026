import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from utils import build_worker_env, get_best_gpu


def main():
    # Lock GPU before any CUDA-capable imports or subprocesses are spawned.
    # Setting os.environ here is inherited by all child subprocesses automatically.
    best_gpu = get_best_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = best_gpu

    parser = argparse.ArgumentParser(description="Phase 2: RVC Audio Generation")
    parser.add_argument("--manifest", type=str, default="data/dataset_manifest.json")
    parser.add_argument("--output_dir", type=str, default="data/generated_audio")
    parser.add_argument("--model", type=str, default="audio/Applio/logs/DonaldTrump/DonaldTrump_475e_8075s.pth")
    parser.add_argument("--index", type=str, default="audio/Applio/logs/DonaldTrump/DonaldTrump.index")
    parser.add_argument("--speaker_name", type=str, default="DonaldTrump")
    parser.add_argument("--dev_mode", action="store_true", help="Single condition (C3/fp32) for fast iteration")
    parser.add_argument("--fp32_only", action="store_true",
                        help="Run all chunk conditions but fp32 precision only (for Alex SQ1 chunk-length research)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}. Run Phase 1 first.")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    videos = manifest.get("videos", {})
    if not videos:
        print("[WARN] No videos in manifest.")
        sys.exit(0)

    if args.dev_mode:
        chunk_conditions = {"C3": 192}
        precisions = ["fp32"]
    elif args.fp32_only:
        # All 5 chunk conditions, fp32 only — for Alex SQ1 (chunk-length impact).
        # Reduces 1500 files to 500 vs a full run.
        chunk_conditions = {"C1": 24, "C2": 72, "C3": 192, "C4": 384, "C5": 768}
        precisions = ["fp32"]
    else:
        chunk_conditions = {"C1": 24, "C2": 72, "C3": 192, "C4": 384, "C5": 768}
        precisions = ["fp32", "fp16", "int8"]

    infer_script = Path("audio/infer_with_timing.py").absolute()
    if not infer_script.exists():
        print(f"[ERROR] Inference script not found: {infer_script}")
        sys.exit(1)

    # Resolved once outside the loop — path is constant for the entire run
    applio_python = str(Path("audio/Applio/.venv/bin/python").absolute())
    if not Path(applio_python).exists():
        print(f"[ERROR] Applio venv not found: {applio_python}. Run setup_workers.py first.")
        sys.exit(1)

    # Build the worker env once: LD_LIBRARY_PATH for Applio's pip-installed CUDA libs
    worker_env = build_worker_env(applio_python)
    # Propagate the selected GPU into the worker subprocess explicitly
    worker_env["CUDA_VISIBLE_DEVICES"] = best_gpu

    total = len(videos) * len(chunk_conditions) * len(precisions)
    current = 0

    for vid_id, data in videos.items():
        source_audio = data.get("source_audio")
        if not source_audio or not Path(source_audio).exists():
            print(f"[WARN] Missing source audio for {vid_id}, skipping.")
            continue

        for cond_name, chunk_size in chunk_conditions.items():
            for prec in precisions:
                current += 1
                base_name = f"{vid_id}_{args.speaker_name}_{cond_name}_{prec}"
                out_wav = output_dir / f"{base_name}.wav"

                if out_wav.exists() and out_wav.stat().st_size > 0:
                    print(f"[{current}/{total}] SKIP {out_wav.name}")
                    continue

                print(f"\n[{current}/{total}] Generating {base_name}...")

                cmd = [
                    applio_python, str(infer_script),
                    "--source", str(source_audio),
                    "--output", str(out_wav),
                    "--model", args.model,
                    "--index", args.index,
                    "--read-chunk-size", str(chunk_size),
                    "--target-speaker", args.speaker_name,
                    "--precision", prec,
                ]

                try:
                    subprocess.run(cmd, env=worker_env, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  [ERROR] RVC inference failed for {base_name}: exit code {e.returncode}")


if __name__ == "__main__":
    main()