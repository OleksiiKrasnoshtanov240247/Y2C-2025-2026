import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from utils import get_best_gpu


# ── helpers ───────────────────────────────────────────────────────────────────

def load_existing_evaluations(output_csv: Path) -> set:
    """
    Return a set of (vid_id, architecture) pairs already in the video CSV.
    Prevents duplicate rows on re-runs.
    """
    evaluated = set()
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return evaluated
    try:
        with open(output_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vid_id = row.get("vid_id", "").strip()
                arc = row.get("architecture", "").strip()
                if vid_id and arc:
                    evaluated.add((vid_id, arc))
        print(f"[INFO] Loaded {len(evaluated)} existing video evaluations from {output_csv.name}")
    except Exception as e:
        print(f"[WARN] Could not read video CSV ({e}). All videos will be re-evaluated.")
    return evaluated


def load_existing_audio_evaluations(audio_csv: Path) -> set:
    """
    Return a set of audio base names already in the audio CSV.
    Key is the wav stem: {vid_id}_{speaker}_{condition}_{precision}
    """
    evaluated = set()
    if not audio_csv.exists() or audio_csv.stat().st_size == 0:
        return evaluated
    try:
        with open(audio_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fname = row.get("transformed_file", "").strip()
                if fname:
                    evaluated.add(Path(fname).stem)
        print(f"[INFO] Loaded {len(evaluated)} existing audio evaluations from {audio_csv.name}")
    except Exception as e:
        print(f"[WARN] Could not read audio CSV ({e}). All audio files will be re-evaluated.")
    return evaluated


def parse_video_stem(stem: str) -> dict | None:
    """
    Parse metadata from a merged video filename.

    Expected format:
      video_{source_id}_{clip_id}_{speaker}_{condition}_{precision}_{architecture}
    Example:
      video_id00328_00092_DonaldTrump_C3_fp32_Deep-Live-Cam
    """
    parts = stem.split("_")
    if len(parts) < 7:
        return None

    return {
        "vid_id": f"{parts[0]}_{parts[1]}_{parts[2]}",
        "target_speaker": parts[3],
        "chunk_condition": next((p for p in parts if p in {"C1", "C2", "C3", "C4", "C5"}), "unknown"),
        "precision": next((p for p in parts if p in {"fp32", "fp16", "int8"}), "unknown"),
        "architecture": parts[-1],
    }


def parse_audio_stem(stem: str) -> dict | None:
    """
    Parse metadata from a generated audio filename.

    Expected format:
      video_{source_id}_{clip_id}_{speaker}_{condition}_{precision}
    Example:
      video_id00328_00092_DonaldTrump_C3_fp32
    """
    parts = stem.split("_")
    if len(parts) < 6:
        return None

    return {
        "vid_id": f"{parts[0]}_{parts[1]}_{parts[2]}",
        "target_speaker": parts[3],
        "chunk_condition": next((p for p in parts if p in {"C1", "C2", "C3", "C4", "C5"}), "unknown"),
        "precision": next((p for p in parts if p in {"fp32", "fp16", "int8"}), "unknown"),
    }


# ── Phase 5a: Video metrics ───────────────────────────────────────────────────

def run_video_metrics(args, manifest_videos: dict, best_gpu: str):
    merged_dir = Path(args.merged_dir)
    output_csv = Path(args.output_csv)
    audio_dir = Path(args.audio_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    video_metrics_script = Path("video_metrics/metrics.py").absolute()
    if not video_metrics_script.exists():
        print(f"[ERROR] Metrics script not found: {video_metrics_script}")
        return

    target_ref = Path("src/assets/target_image/elunma.jpeg").absolute()
    if not target_ref.exists():
        print(f"[ERROR] Target reference image not found: {target_ref}")
        return

    final_videos = sorted(merged_dir.glob("*.mp4"))
    print(f"\n--- Video Metrics ---")
    print(f"Found {len(final_videos)} videos in {merged_dir}")

    already_evaluated = load_existing_evaluations(output_csv)

    csv_headers = [
        "vid_id", "target_speaker", "chunk_condition", "precision", "architecture",
        "identity_arcface_mean", "landmark_nme_mean",
        "temporal_lpips_mean", "fid_score", "latency_p95_ms",
    ]

    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with open(output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        if write_header:
            writer.writeheader()

        for video_path in final_videos:
            meta = parse_video_stem(video_path.stem)
            if meta is None:
                print(f"  [WARN] Cannot parse filename: {video_path.name}, skipping.")
                continue

            vid_id = meta["vid_id"]
            arc = meta["architecture"]

            if (vid_id, arc) in already_evaluated:
                print(f"  [SKIP] Already evaluated: {vid_id} / {arc}")
                continue

            source_video = manifest_videos.get(vid_id, {}).get("source_video")
            if not source_video:
                print(f"  [WARN] {vid_id} not in current manifest (stale file?), skipping.")
                continue
            if not Path(source_video).exists():
                print(f"  [WARN] Source video missing on disk: {source_video}")
                continue

            print(f"  [EVAL] {arc} / {vid_id}")

            tmp_json = video_path.parent / f"{video_path.stem}_metrics.json"

            audio_base = video_path.stem.rsplit("_", 1)[0]
            perf_json = audio_dir / f"{audio_base}.perf.json"

            gpu_env = os.environ.copy()
            gpu_env["CUDA_VISIBLE_DEVICES"] = best_gpu
            gpu_env["TF_CPP_MIN_LOG_LEVEL"] = "3"
            gpu_env["GLOG_minloglevel"] = "3"
            gpu_env["TF_ENABLE_ONEDNN_OPTS"] = "0"
            gpu_env["ORT_LOGGING_LEVEL"] = "4"

            cmd = [
                sys.executable, str(video_metrics_script),
                "--generated", str(video_path.absolute()),
                "--reference", str(target_ref),
                "--source", str(Path(source_video).absolute()),
                "--out", str(tmp_json.absolute()),
            ]
            if perf_json.exists():
                cmd += ["--perf_json", str(perf_json.absolute())]

            try:
                subprocess.run(
                    cmd,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    env=gpu_env,
                    check=True,
                )

                if not tmp_json.exists():
                    print(f"  [WARN] Metrics script wrote no JSON for {video_path.name}")
                    continue

                with open(tmp_json, "r", encoding="utf-8") as jf:
                    metrics = json.load(jf).get("metrics", {})

                row = {
                    "vid_id": vid_id,
                    "target_speaker": meta["target_speaker"],
                    "chunk_condition": meta["chunk_condition"],
                    "precision": meta["precision"],
                    "architecture": arc,
                    "identity_arcface_mean": metrics.get("identity_arcface", {}).get("mean"),
                    "landmark_nme_mean": metrics.get("landmark_nme", {}).get("mean_nme"),
                    "temporal_lpips_mean": metrics.get("temporal_lpips", {}).get("mean"),
                    "fid_score": metrics.get("fid", {}).get("fid"),
                    "latency_p95_ms": metrics.get("latency", {}).get("p95_latency_ms"),
                }
                writer.writerow(row)
                csvfile.flush()

            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] Video metrics failed for {video_path.name}: exit code {e.returncode}")
            except Exception as e:
                print(f"  [ERROR] Unexpected error for {video_path.name}: {e}")
            finally:
                if tmp_json.exists():
                    tmp_json.unlink()

    print(f"Video metrics -> {output_csv}")


# ── Phase 5b: Audio metrics ───────────────────────────────────────────────────

def load_transcript_cache(cache_path: Path) -> dict:
    """Load cached ground-truth Whisper transcripts keyed by vid_id."""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"[INFO] Loaded {len(cache)} cached Whisper transcripts from {cache_path.name}")
            return cache
        except Exception as e:
            print(f"[WARN] Could not read transcript cache ({e}). Will re-transcribe as needed.")
    return {}


def save_transcript_cache(cache: dict, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def run_audio_metrics(args, manifest_videos: dict, best_gpu: str):
    """
    Run audio evaluation (WER/CER, STOI, PESQ, SNR, MCD, SECS, F0) on each
    generated audio file.

    Audio metrics are per (vid_id, condition, precision) — independent of
    video architecture.  Key optimisations over a naive implementation:

    1. Transcript caching — evaluate.py transcribes the original audio with
       Whisper to get a ground-truth reference, then transcribes the converted
       audio and computes WER.  In a full run (100 videos × 5 conditions × 3
       precisions = 1,500 files) the original audio is the SAME for all 15
       conditions/precisions of the same vid_id.  Without caching, Whisper
       runs 1,500 times on originals.  With caching it runs 100 times on
       originals (once per vid_id) and 1,500 times on converted files only.
       Saving: 1,400 Whisper calls ≈ 7–23 hours at 30s/call.

    2. Whisper on GPU — evaluate.py accepts --whisper-device.  cuda int8 is
       ~8x faster than cpu int8 for large-v3-turbo.  Passed from orchestrator.

    evaluate.py must run with cwd=audio_testing_dir because it uses relative
    imports (from metrics.speaker_similarity import ...).
    """
    audio_dir = Path(args.audio_dir)
    audio_output_csv = Path(args.audio_output_csv)
    audio_testing_dir = Path(args.audio_testing_dir).absolute()
    transcript_cache_path = Path(args.transcript_cache)
    audio_output_csv.parent.mkdir(parents=True, exist_ok=True)

    audio_eval_script = audio_testing_dir / "evaluate.py"
    if not audio_eval_script.exists():
        print(f"[WARN] Audio evaluation script not found: {audio_eval_script}. Skipping audio metrics.")
        return

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"[WARN] No wav files in {audio_dir}. Skipping audio metrics.")
        return

    print(f"\n--- Audio Metrics ---")
    print(f"Found {len(wav_files)} audio files in {audio_dir}")

    already_evaluated = load_existing_audio_evaluations(audio_output_csv)
    transcript_cache = load_transcript_cache(transcript_cache_path)

    audio_env = os.environ.copy()
    audio_env["CUDA_VISIBLE_DEVICES"] = best_gpu
    audio_env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    audio_env["GLOG_minloglevel"] = "3"

    cache_dirty = False

    for wav_path in wav_files:
        meta = parse_audio_stem(wav_path.stem)
        if meta is None:
            print(f"  [WARN] Cannot parse audio filename: {wav_path.name}, skipping.")
            continue

        if wav_path.stem in already_evaluated:
            print(f"  [SKIP] Already evaluated: {wav_path.stem}")
            continue

        vid_id = meta["vid_id"]
        source_audio = manifest_videos.get(vid_id, {}).get("source_audio")
        if not source_audio:
            print(f"  [WARN] {vid_id} not in manifest (stale file?), skipping.")
            continue
        if not Path(source_audio).exists():
            print(f"  [WARN] Source audio missing: {source_audio}")
            continue

        print(f"  [EVAL] {wav_path.stem}")

        cmd = [
            sys.executable, str(audio_eval_script),
            "--original",    str(Path(source_audio).absolute()),
            "--transformed", str(wav_path.absolute()),
            "--output",      str(audio_output_csv.absolute()),
            "--whisper-device",       args.whisper_device,
            "--whisper-compute-type", "int8",
            "--whisper-model",        "large-v3-turbo",
        ]

        # Pass cached transcript if available for this vid_id.
        # This skips the Whisper call on the original audio — the single
        # biggest time cost when running many conditions per video.
        if vid_id in transcript_cache:
            cmd += ["--transcript", transcript_cache[vid_id]]
            print(f"  [CACHE] Using cached transcript for {vid_id}")

        try:
            subprocess.run(
                cmd,
                cwd=str(audio_testing_dir),
                env=audio_env,
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True,
            )

            # After successful evaluation, read the CSV to extract the transcript
            # evaluate.py wrote and cache it for future conditions of this vid_id.
            if vid_id not in transcript_cache:
                try:
                    import csv as _csv
                    with open(audio_output_csv, "r", encoding="utf-8") as _f:
                        rows = list(_csv.DictReader(_f))
                    for row in reversed(rows):
                        # Most recently written row for this original file
                        if Path(row.get("original_file", "")).stem == Path(source_audio).stem:
                            transcript = row.get("ground_truth_transcript", "").strip()
                            if transcript:
                                transcript_cache[vid_id] = transcript
                                cache_dirty = True
                                print(f"  [CACHE] Stored transcript for {vid_id}")
                            break
                except Exception as e:
                    print(f"  [WARN] Could not extract transcript from CSV: {e}")

        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Audio evaluation failed for {wav_path.name}: exit code {e.returncode}")
        except Exception as e:
            print(f"  [ERROR] Unexpected error for {wav_path.name}: {e}")

        # Persist cache after each successful eval so a crash doesn't lose progress
        if cache_dirty:
            save_transcript_cache(transcript_cache, transcript_cache_path)
            cache_dirty = False

    print(f"Audio metrics -> {audio_output_csv}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    best_gpu = get_best_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = best_gpu

    parser = argparse.ArgumentParser(description="Phase 5: Metrics Evaluation")
    # Video metrics args
    parser.add_argument("--manifest",    type=str, default="data/dataset_manifest.json")
    parser.add_argument("--merged_dir",  type=str, default="data/final_merged")
    parser.add_argument("--audio_dir",   type=str, default="data/generated_audio",
                        help="Generated audio directory (also used for perf.json sidecars)")
    parser.add_argument("--output_csv",  type=str, default="results/master_evaluation_results.csv")
    # Audio metrics args
    parser.add_argument("--processed_audio_dir", type=str, default="data/processed_audio",
                        help="Source audio directory (pre-RVC extracted WAVs)")
    parser.add_argument("--audio_testing_dir",   type=str, default="audio/audio_testing",
                        help="Directory containing evaluate.py and metrics/ subpackage")
    parser.add_argument("--audio_output_csv",    type=str, default="results/audio_evaluation_results.csv",
                        help="Output CSV for audio metrics")
    parser.add_argument("--transcript_cache",    type=str, default="results/whisper_transcript_cache.json",
                        help="JSON file caching ground-truth Whisper transcripts per vid_id")
    parser.add_argument("--whisper_device",      type=str, default="cuda", choices=["cpu", "cuda", "auto"],
                        help="Device for Whisper transcription (cuda is ~8x faster for int8)")
    parser.add_argument("--audio_only",          action="store_true",
                        help="Skip video metrics entirely, run audio metrics only")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    merged_dir    = Path(args.merged_dir)

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)
    if not merged_dir.exists():
        print(f"[ERROR] Merged video directory not found: {merged_dir}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_videos = manifest.get("videos", {})

    print(f"\n--- Starting Metrics Evaluation ---")

    if not getattr(args, "audio_only", False):
        run_video_metrics(args, manifest_videos, best_gpu)
    else:
        print("[INFO] --audio_only: skipping video metrics")

    run_audio_metrics(args, manifest_videos, best_gpu)

    print(f"\nPhase 5 complete.")
    print(f"Video results : {args.output_csv}")
    print(f"Audio results : {args.audio_output_csv}")


if __name__ == "__main__":
    main()