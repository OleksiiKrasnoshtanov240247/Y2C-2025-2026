"""
Batch orchestrator for the baseline experiment.

Runs infer_with_timing + evaluate for all combinations of:
  source clips  x  target speakers  x  repetitions

Saves all results to a single CSV file (results/baseline_raw.csv).

Directory structure expected:
    audio/
    ├── assets/
    │   ├── input/              # 30 source clips (.wav)
    │   ├── targets/            # 1 reference clip per target speaker (.wav)
    │   └── output/             # auto-created per run
    ├── models/
    │   └── rvc/
    │       └── <speaker_name>/
    │           ├── <speaker_name>.pth
    │           └── <speaker_name>.index
    └── results/
        └── baseline_raw.csv    # output

Usage:
    python run_baseline.py
    python run_baseline.py --reps 3 --condition C3
    python run_baseline.py --dry-run
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = SCRIPT_DIR  # audio/

# Chunk length conditions from proposal Table 1
CONDITIONS = {
    "C1": {"read_chunk_size": 24,  "block_frame_ms": 64},
    "C2": {"read_chunk_size": 72,  "block_frame_ms": 192},
    "C3": {"read_chunk_size": 192, "block_frame_ms": 512},   # default / baseline
    "C4": {"read_chunk_size": 384, "block_frame_ms": 1024},
    "C5": {"read_chunk_size": 768, "block_frame_ms": 2048},
}


def discover_source_clips(input_dir: Path) -> list[Path]:
    """Find all audio source clips in the input directory."""
    clips = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".wav", ".ogg", ".flac", ".mp3")
    )
    return clips


def discover_target_speakers(
    models_dir: Path, targets_dir: Path
) -> list[dict]:
    """Discover target speakers from models/rvc/ and assets/targets/.

    Each target speaker needs:
    - A .pth model file in models/rvc/<speaker>/
    - A .index file in models/rvc/<speaker>/
    - A reference audio in assets/targets/<speaker>.wav

    Returns list of dicts with keys: name, model_path, index_path, reference_path
    """
    speakers = []

    if not models_dir.is_dir():
        return speakers

    for speaker_dir in sorted(models_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue

        name = speaker_dir.name

        # Find .pth model file
        pth_files = list(speaker_dir.glob("*.pth"))
        if not pth_files:
            print(f"  WARNING: No .pth file in {speaker_dir}, skipping {name}")
            continue
        model_path = pth_files[0]

        # Find .index file
        index_files = list(speaker_dir.glob("*.index"))
        if not index_files:
            print(f"  WARNING: No .index file in {speaker_dir}, skipping {name}")
            continue
        index_path = index_files[0]

        # Find reference audio in assets/targets/
        ref_path = None
        for ext in (".wav", ".flac", ".ogg", ".mp3"):
            candidate = targets_dir / f"{name}{ext}"
            if candidate.is_file():
                ref_path = candidate
                break

        if ref_path is None:
            print(f"  WARNING: No reference audio for '{name}' in {targets_dir}")
            print(f"           SECS will compare original vs converted (not target)")

        speakers.append({
            "name": name,
            "model_path": str(model_path),
            "index_path": str(index_path),
            "reference_path": str(ref_path) if ref_path else None,
        })

    return speakers


def _known_speakers(audio_dir: Path) -> list[dict]:
    """Return speaker entries for this project's actual Applio model layout.

    The Applio/logs/ directory uses inconsistent naming and structure,
    so auto-discovery from models/rvc/ won't work. This defines the
    actual paths for each speaker with a trained model and reference audio.
    """
    logs = audio_dir / "Applio" / "logs"
    originals = audio_dir / "assets" / "original"

    entries = [
        {
            "name": "DonaldTrump",
            "model_path": logs / "DonaldTrump" / "DonaldTrump_475e_8075s.pth",
            "index_path": logs / "DonaldTrump" / "DonaldTrump.index",
            "reference_path": originals / "trump.mp3",
        },
        {
            "name": "GeorgeBanks",
            "model_path": logs / "GeorgeBanks" / "George Banks_500e_41500s.pth",
            "index_path": logs / "GeorgeBanks" / "George Banks.index",
            "reference_path": originals / "banks.wav",
        },
        {
            "name": "Wheatley-HD",
            "model_path": logs / "Wheatley-HD_e450_s40050.pth",
            "index_path": logs / "added_IVF5119_Flat_nprobe_1_Wheatley-HD_v2.index",
            "reference_path": originals / "wheatly.wav",
        },
    ]

    valid = []
    for e in entries:
        ok = True
        for key in ("model_path", "index_path"):
            if not e[key].is_file():
                print(f"  WARNING: {e['name']} missing {key}: {e[key]}")
                ok = False
        if e["reference_path"] and not e["reference_path"].is_file():
            print(f"  WARNING: {e['name']} missing reference: {e['reference_path']}")
            e["reference_path"] = None
        if ok:
            valid.append({
                "name": e["name"],
                "model_path": str(e["model_path"]),
                "index_path": str(e["index_path"]),
                "reference_path": str(e["reference_path"]) if e["reference_path"] else None,
            })
        else:
            print(f"  WARNING: Skipping {e['name']} (missing model files)")

    return valid


def run_single(
    source_path: str,
    target: dict,
    rep: int,
    condition: str,
    output_dir: Path,
    results_writer,
    evaluate_fn,
    infer_fn,
) -> dict | None:
    """Run inference + evaluation for a single combination."""
    source_name = Path(source_path).stem
    speaker_name = target["name"]
    rcs = CONDITIONS[condition]["read_chunk_size"]

    # Output paths
    out_name = f"{source_name}__{speaker_name}__{condition}__rep{rep}"
    out_wav = output_dir / f"{out_name}.wav"
    out_perf = output_dir / f"{out_name}.perf.json"

    print(f"\n  [{source_name}] -> [{speaker_name}] | {condition} | rep {rep}")

    try:
        # Step 1: Inference with timing
        perf_stats = infer_fn(
            source_path=source_path,
            output_path=str(out_wav),
            model_path=target["model_path"],
            index_path=target["index_path"],
            read_chunk_size=rcs,
            target_speaker=speaker_name,
        )

        # Step 2: Evaluation
        eval_results = evaluate_fn(
            original_path=source_path,
            transformed_path=str(out_wav),
            target_path=target["reference_path"],
        )

        # Step 3: Merge into one row
        row = {
            "source_clip": source_name,
            "target_speaker": speaker_name,
            "repetition": rep,
            "condition": condition,
            "read_chunk_size": rcs,
            "block_frame_ms": CONDITIONS[condition]["block_frame_ms"],
        }

        # Add evaluation metrics
        metric_keys = [
            "conversion_secs", "content_wer", "content_cer",
            "quality_stoi", "quality_pesq", "quality_snr_db",
            "conversion_mcd_db", "prosody_f0_correlation", "prosody_f0_rmse_hz",
        ]
        for k in metric_keys:
            row[k] = eval_results.get(k)

        # Add perf stats
        row["mean_chunk_latency_ms"] = perf_stats.get("mean_chunk_latency_ms")
        row["min_chunk_latency_ms"] = perf_stats.get("min_chunk_latency_ms")
        row["max_chunk_latency_ms"] = perf_stats.get("max_chunk_latency_ms")
        row["std_chunk_latency_ms"] = perf_stats.get("std_chunk_latency_ms")
        row["total_latency_ms"] = perf_stats.get("total_latency_ms")
        row["num_chunks"] = perf_stats.get("num_chunks")
        row["peak_gpu_memory_mb"] = perf_stats.get("peak_gpu_memory_mb")

        # Write row to CSV
        results_writer.writerow(row)

        return row

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch orchestrator for baseline experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Conditions available:
  C1:  ~64 ms   (read_chunk_size=24)
  C2: ~192 ms   (read_chunk_size=72)
  C3: ~512 ms   (read_chunk_size=192)  <- baseline default
  C4: ~1024 ms  (read_chunk_size=384)
  C5: ~2048 ms  (read_chunk_size=768)

Example:
  python run_baseline.py                         # full baseline (C3, 5 reps)
  python run_baseline.py --reps 1 --dry-run      # preview without running
  python run_baseline.py --condition C1 C2 C3    # multiple conditions
        """,
    )
    parser.add_argument(
        "--reps", type=int, default=5,
        help="Number of repetitions per source-target pair (default: 5)",
    )
    parser.add_argument(
        "--condition", nargs="+", default=["C3"],
        choices=list(CONDITIONS.keys()),
        help="Chunk length condition(s) to run (default: C3)",
    )
    parser.add_argument(
        "--input-dir", type=str, default=None, dest="input_dir",
        help="Directory with source clips (default: assets/input)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, dest="output_dir",
        help="Directory for converted outputs (default: assets/output)",
    )
    parser.add_argument(
        "--results-csv", type=str, default=None, dest="results_csv",
        help="Output CSV path (default: results/baseline_raw.csv)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Print what would be run without actually running inference",
    )

    args = parser.parse_args()

    # Resolve directories
    input_dir = Path(args.input_dir) if args.input_dir else AUDIO_DIR / "assets" / "input"
    output_dir = Path(args.output_dir) if args.output_dir else AUDIO_DIR / "assets" / "output"
    models_dir = AUDIO_DIR / "models" / "rvc"
    targets_dir = AUDIO_DIR / "assets" / "targets"
    results_dir = AUDIO_DIR / "results"
    results_csv = Path(args.results_csv) if args.results_csv else results_dir / "baseline_raw.csv"

    print("=" * 60)
    print("  Baseline Experiment Orchestrator")
    print("=" * 60)
    print(f"  Input dir    : {input_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Models dir   : {models_dir}")
    print(f"  Targets dir  : {targets_dir}")
    print(f"  Results CSV  : {results_csv}")
    print(f"  Conditions   : {', '.join(args.condition)}")
    print(f"  Repetitions  : {args.reps}")
    print("-" * 60)

    # Discover data
    source_clips = discover_source_clips(input_dir)
    target_speakers = _known_speakers(AUDIO_DIR)
    if not target_speakers:
        target_speakers = discover_target_speakers(models_dir, targets_dir)

    if not source_clips:
        print(f"\n  ERROR: No source clips found in {input_dir}")
        print(f"  Place .wav files in {input_dir} and re-run.")
        sys.exit(1)

    if not target_speakers:
        print(f"\n  ERROR: No target speakers found in {models_dir}")
        print(f"  Expected structure:")
        print(f"    {models_dir}/<speaker_name>/<speaker_name>.pth")
        print(f"    {models_dir}/<speaker_name>/<speaker_name>.index")
        sys.exit(1)

    total_runs = len(source_clips) * len(target_speakers) * args.reps * len(args.condition)

    print(f"\n  Source clips  : {len(source_clips)}")
    for c in source_clips:
        print(f"    - {c.name}")
    print(f"  Target speakers: {len(target_speakers)}")
    for t in target_speakers:
        ref = "yes" if t["reference_path"] else "NO"
        print(f"    - {t['name']} (ref audio: {ref})")
    print(f"  Total runs   : {total_runs}")

    if args.dry_run:
        print("\n  DRY RUN — no inference will be executed.")
        print(f"  Would generate {total_runs} rows in {results_csv}")
        return

    # Import the actual functions (deferred so --dry-run works without deps)
    sys.path.insert(0, str(AUDIO_DIR / "audio_testing"))
    from infer_with_timing import infer_with_timing
    from evaluate import evaluate

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # CSV setup
    csv_fieldnames = [
        "source_clip", "target_speaker", "repetition",
        "condition", "read_chunk_size", "block_frame_ms",
        "conversion_secs", "content_wer", "content_cer",
        "quality_stoi", "quality_pesq", "quality_snr_db",
        "conversion_mcd_db", "prosody_f0_correlation", "prosody_f0_rmse_hz",
        "mean_chunk_latency_ms", "min_chunk_latency_ms", "max_chunk_latency_ms",
        "std_chunk_latency_ms", "total_latency_ms", "num_chunks",
        "peak_gpu_memory_mb",
    ]

    # Append mode: continue from where we left off if CSV exists
    file_exists = results_csv.is_file()
    csv_file = open(results_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    if not file_exists:
        writer.writeheader()

    completed = 0
    t_start = time.time()

    try:
        for condition in args.condition:
            for source_path in source_clips:
                for target in target_speakers:
                    for rep in range(1, args.reps + 1):
                        run_single(
                            source_path=str(source_path),
                            target=target,
                            rep=rep,
                            condition=condition,
                            output_dir=output_dir,
                            results_writer=writer,
                            evaluate_fn=evaluate,
                            infer_fn=infer_with_timing,
                        )
                        csv_file.flush()
                        completed += 1

                        elapsed = time.time() - t_start
                        rate = elapsed / completed if completed else 0
                        remaining = rate * (total_runs - completed)
                        print(f"  Progress: {completed}/{total_runs} "
                              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    except KeyboardInterrupt:
        print(f"\n  Interrupted after {completed}/{total_runs} runs.")
    finally:
        csv_file.close()

    print("\n" + "=" * 60)
    print(f"  Experiment complete: {completed}/{total_runs} runs")
    print(f"  Results saved to: {results_csv}")
    print(f"  Total time: {time.time() - t_start:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
