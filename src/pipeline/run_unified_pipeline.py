import argparse
<<<<<<< HEAD
import os
=======
import random
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
import subprocess
import sys
from pathlib import Path

<<<<<<< HEAD
def run_phase(phase_script: Path, description: str, *args) -> bool:
    print(f"\n{'='*70}")
    print(f"  Executing {description}...")
    print(f"{'='*70}")
    
    cmd = [sys.executable, str(phase_script)] + list(args)
    try:
        # Run process interactively to show progress
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] {description} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[FATAL ERROR] Exception during {description}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator: Unified Deepfake Evaluation Pipeline")
    parser.add_argument("--skip_p1", action="store_true", help="Skip Phase 1: Data Preparation")
    parser.add_argument("--skip_p2", action="store_true", help="Skip Phase 2: Audio Generation")
    parser.add_argument("--skip_p3", action="store_true", help="Skip Phase 3: Video Generation")
    parser.add_argument("--skip_p4", action="store_true", help="Skip Phase 4: AV Merging")
    parser.add_argument("--skip_p5", action="store_true", help="Skip Phase 5: Metrics Evaluation")
    
    # Global overrides
    parser.add_argument("--model", type=str, default="audio/Applio/logs/DonaldTrump/DonaldTrump_475e_8075s.pth", help="Target RVC Model path")
    parser.add_argument("--index", type=str, default="audio/Applio/logs/DonaldTrump/DonaldTrump.index", help="Target RVC Index path")
    parser.add_argument("--dev_mode", action="store_true", help="Run in dev mode (2 videos, simplified configurations)")
    
    args = parser.parse_args()
    
    pipeline_dir = Path(__file__).parent.absolute()
    
    p1_script = pipeline_dir / "1_prepare_data.py"
    p2_script = pipeline_dir / "2_generate_audio.py"
    p3_script = pipeline_dir / "3_generate_video.py"
    p4_script = pipeline_dir / "4_merge_av.py"
    p5_script = pipeline_dir / "5_evaluate_metrics.py"
    
    # Check scripts exist
    for script in [p1_script, p2_script, p3_script, p4_script, p5_script]:
        if not script.exists():
            print(f"[FATAL] Cannot locate pipeline step: {script.name}")
            sys.exit(1)
    
    print("="*70)
    print("  Group O1 - Unified Deepfake Evaluation Pipeline started")
    print("="*70)
    
    if not args.skip_p1:
        p1_args = []
        if args.dev_mode:
            p1_args.extend(["--num_samples", "2"])
        if not run_phase(p1_script, "Phase 1: Data Preparation", *p1_args): sys.exit(1)
        
    if not args.skip_p2:
        p2_args = ["--model", args.model, "--index", args.index]
        if args.dev_mode:
            p2_args.append("--dev_mode")
        if not run_phase(p2_script, "Phase 2: RVC Audio Generation", *p2_args): sys.exit(1)
                         
    if not args.skip_p3:
        if not run_phase(p3_script, "Phase 3: Deepfake Video Generation"): sys.exit(1)
        
    if not args.skip_p4:
        if not run_phase(p4_script, "Phase 4: Audio-Video FFMPEG Merging"): sys.exit(1)
        
    if not args.skip_p5:
        if not run_phase(p5_script, "Phase 5: Unified Metrics Evaluation"): sys.exit(1)
        
    print("\n" + "="*70)
    print("  PIPELINE EXECUTION COMPLETE - ALL PHASES SUCCESSFUL")
    print("="*70)

if __name__ == "__main__":
    main()
=======

def run_phase(script: Path, description: str, extra_args: list) -> bool:
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    cmd = [sys.executable, str(script)] + extra_args
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] {description} exited with code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[FATAL] {description} raised an exception: {e}")
        return False


def resolve_paths(run_id: str) -> dict:
    """
    All output paths scoped under data/runs/{run_id}/ when run_id is set.
    This prevents dev, global_test, and full runs from overwriting each other.
    """
    if run_id:
        base = Path("data/runs") / run_id
        return {
            "manifest":            str(base / "dataset_manifest.json"),
            "audio_dir":           str(base / "generated_audio"),
            "processed_audio_dir": str(base / "processed_audio"),
            "video_dir":           str(base / "generated_video"),
            "merged_dir":          str(base / "final_merged"),
            "results_csv":         str(Path("results") / run_id / "master_evaluation_results.csv"),
            "audio_results_csv":   str(Path("results") / run_id / "audio_evaluation_results.csv"),
            "transcript_cache":    str(Path("results") / run_id / "whisper_transcript_cache.json"),
        }
    return {
        "manifest":            "data/dataset_manifest.json",
        "audio_dir":           "data/generated_audio",
        "processed_audio_dir": "data/processed_audio",
        "video_dir":           "data/generated_video",
        "merged_dir":          "data/final_merged",
        "results_csv":         "results/master_evaluation_results.csv",
        "audio_results_csv":   "results/audio_evaluation_results.csv",
        "transcript_cache":    "results/whisper_transcript_cache.json",
    }


def print_mode_summary(args):
    print("="*70)
    print("  Group O1 — Unified Deepfake Evaluation Pipeline")
    if args.run_id:
        print(f"  Run ID       : {args.run_id}")
    if args.dev_mode:
        print("  Mode         : DEV  (2 random videos, C3/fp32 only)")
    elif args.audio_only:
        print("  Mode         : AUDIO ONLY  (Phase 2 + audio metrics, skip video)")
    elif args.global_test:
        print(f"  Mode         : GLOBAL TEST  ({args.num_samples} videos, all conditions, all precisions)")
        print(f"  Seed         : {args.seed}  (fixed for reproducibility)")
    else:
        print("  Mode         : FULL  (all conditions, all precisions)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Unified Deepfake Evaluation Pipeline — Master Orchestrator")
    parser.add_argument("--skip_p1", action="store_true")
    parser.add_argument("--skip_p2", action="store_true")
    parser.add_argument("--skip_p3", action="store_true")
    parser.add_argument("--skip_p4", action="store_true")
    parser.add_argument("--skip_p5", action="store_true")
    parser.add_argument("--model", type=str, default="audio/Applio/logs/DonaldTrump/DonaldTrump_475e_8075s.pth")
    parser.add_argument("--index", type=str, default="audio/Applio/logs/DonaldTrump/DonaldTrump.index")

    # ── Run mode flags (mutually exclusive) ──────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dev_mode", action="store_true",
        help="2 random videos, C3/fp32 only. Randomised seed each run for variety.",
    )
    mode_group.add_argument(
        "--audio_only", action="store_true",
        help=(
            "Skip video generation and video metrics entirely. "
            "Runs Phase 1 (data prep), Phase 2 (audio generation, skipping existing files), "
            "and Phase 5 audio metrics only. "
            "Use this to process Alex/Aron audio conditions in parallel while video runs elsewhere."
        ),
    )
    mode_group.add_argument(
        "--global_test", action="store_true",
        help=(
            "Full research run: all 5 chunk conditions, all 3 precisions, all architectures. "
            "Covers all sub-questions (Alex/Aron: audio conditions+precisions; "
            "Filipp/Maksym: all architectures; Danil: source material for MOS study). "
            "Stratified diversity sampling across race/gender groups. "
            "Fixed seed 42 for reproducibility. Default 10 videos; use --num_samples to change."
        ),
    )

    # ── Sample size and seed controls ────────────────────────────────────────
    parser.add_argument(
        "--num_samples", type=int, default=0,
        help=(
            "Number of videos to sample from FakeAVCeleb. "
            "Defaults: dev_mode=2, global_test=20, full=100. "
            "Override by passing this flag explicitly."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset sampling (default: 42). dev_mode ignores this.",
    )

    # ── Whisper device for audio metrics ─────────────────────────────────────
    parser.add_argument(
        "--whisper_device", type=str, default="cuda", choices=["cpu", "cuda", "auto"],
        help=(
            "Device for Whisper transcription in Phase 5 audio metrics. "
            "cuda is ~8x faster than cpu for int8 Whisper. Default: cuda. "
            "Fall back to cpu if GPU is occupied by concurrent generation."
        ),
    )

    parser.add_argument(
        "--run_id", type=str, default="",
        help="Optional tag for output directory isolation (e.g. 'global_test_v1').",
    )
    args = parser.parse_args()

    # ── Resolve effective sample count ───────────────────────────────────────
    if args.num_samples > 0:
        effective_samples = args.num_samples          # explicit override always wins
    elif args.dev_mode:
        effective_samples = 2
    elif args.global_test or args.audio_only:
        effective_samples = 10
    else:
        effective_samples = 100                        # full production run

    # ── Auto run_id for global_test to avoid polluting default dirs ──────────
    if args.global_test and not args.run_id:
        args.run_id = "global_test"

    pipeline_dir = Path(__file__).parent.absolute()
    scripts = {
        "p1": pipeline_dir / "1_prepare_data.py",
        "p2": pipeline_dir / "2_generate_audio.py",
        "p3": pipeline_dir / "3_generate_video.py",
        "p4": pipeline_dir / "4_merge_av.py",
        "p5": pipeline_dir / "5_evaluate_metrics.py",
    }
    for name, script in scripts.items():
        if not script.exists():
            print(f"[FATAL] Pipeline script not found: {script.name}")
            sys.exit(1)

    # audio_only forces skip of video phases and isolates to global_test dir by default
    if args.audio_only:
        if not args.run_id:
            args.run_id = "global_test"
        args.skip_p3 = True
        args.skip_p4 = True

    paths = resolve_paths(args.run_id)
    print_mode_summary(args)

    # ── Phase 1: Data Preparation ─────────────────────────────────────────────
    if not args.skip_p1:
        p1_args = [
            "--manifest_path", paths["manifest"],
            "--output_audio_dir", paths["processed_audio_dir"],
            "--num_samples", str(effective_samples),
        ]
        if args.dev_mode:
            # Randomise seed so each dev run picks different videos
            dev_seed = random.randint(0, 2**31 - 1)
            print(f"  Dev seed      : {dev_seed}")
            p1_args += ["--seed", str(dev_seed)]
        else:
            # global_test and full runs use a fixed seed for reproducibility
            p1_args += ["--seed", str(args.seed)]
        if not run_phase(scripts["p1"], "Phase 1: Data Preparation", p1_args):
            sys.exit(1)

    # ── Phase 2: Audio Generation ─────────────────────────────────────────────
    # dev_mode: C3/fp32 only
    # global_test / full: all 5 conditions × all 3 precisions (no --dev_mode flag)
    if not args.skip_p2:
        p2_args = [
            "--manifest",   paths["manifest"],
            "--output_dir", paths["audio_dir"],
            "--model",      args.model,
            "--index",      args.index,
        ]
        if args.dev_mode:
            p2_args.append("--dev_mode")
        if args.audio_only:
            # Alex's SQ1 only needs fp32 across all chunk conditions.
            # Precision sweep (fp16/int8) is Aron's SQ5 — skip it here
            # to reduce 1500 audio files to 500 (10 videos × 5 conditions).
            p2_args.append("--fp32_only")
        if not run_phase(scripts["p2"], "Phase 2: RVC Audio Generation", p2_args):
            sys.exit(1)

    # ── Phase 3: Video Generation ─────────────────────────────────────────────
    # All architectures, all conditions (audio files drive the condition sweep).
    if not args.skip_p3:
        p3_args = [
            "--manifest",   paths["manifest"],
            "--audio_dir",  paths["audio_dir"],
            "--output_dir", paths["video_dir"],
        ]
        if not run_phase(scripts["p3"], "Phase 3: Video Generation", p3_args):
            sys.exit(1)

    # ── Phase 4: AV Merging ───────────────────────────────────────────────────
    if not args.skip_p4:
        p4_args = [
            "--video_dir",  paths["video_dir"],
            "--audio_dir",  paths["audio_dir"],
            "--output_dir", paths["merged_dir"],
        ]
        if not run_phase(scripts["p4"], "Phase 4: AV Merging", p4_args):
            sys.exit(1)

    # ── Phase 5: Metrics Evaluation ───────────────────────────────────────────
    if not args.skip_p5:
        p5_args = [
            "--manifest",            paths["manifest"],
            "--merged_dir",          paths["merged_dir"],
            "--audio_dir",           paths["audio_dir"],
            "--processed_audio_dir", paths["processed_audio_dir"],
            "--output_csv",          paths["results_csv"],
            "--audio_output_csv",    paths["audio_results_csv"],
            "--audio_testing_dir",   "audio/audio_testing",
            "--transcript_cache",    paths["transcript_cache"],
            "--whisper_device",      args.whisper_device,
        ]
        if args.audio_only:
            p5_args.append("--audio_only")
        if not run_phase(scripts["p5"], "Phase 5: Metrics Evaluation", p5_args):
            sys.exit(1)

    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    if args.global_test:
        print(f"\n  Results:")
        print(f"    Video : {paths['results_csv']}")
        print(f"    Audio : {paths['audio_results_csv']}")


if __name__ == "__main__":
    main()
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
