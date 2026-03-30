import argparse
import os
import subprocess
import sys
from pathlib import Path

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
