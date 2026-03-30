import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Unified Metrics Evaluation Pipeline")
    parser.add_argument("--manifest", type=str, default="data/dataset_manifest.json", help="Path to input manifest")
    parser.add_argument("--merged_dir", type=str, default="data/final_merged", help="Directory with final merged AV files")
    parser.add_argument("--output_csv", type=str, default="results/master_evaluation_results.csv", help="Master output for PCA / Correlation")
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    merged_dir = Path(args.merged_dir)
    output_csv = Path(args.output_csv)
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists() or not merged_dir.exists():
        print(f"[ERROR] Required files/directories missing. Run previous phases.")
        sys.exit(1)
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    videos = manifest.get("videos", {})
    if not videos:
        print("[WARN] No videos found in manifest.")
        return
        
    final_videos = list(merged_dir.glob("*.mp4"))
    
    print("\n--- Starting Unified Metrics Evaluation ---")
    print(f"Found {len(final_videos)} generated videos.")
    
    # Locate metrics scripts
    video_metrics_script = Path("video_metrics/metrics.py").absolute()
    audio_metrics_script = Path("audio/audio_testing/evaluate.py").absolute()
    
    # We will accumulate the JSON outputs from the video tools into a master CSV
    csv_headers = [
        "vid_id", "target_speaker", "chunk_condition", "precision", "architecture",
        "identity_arcface_mean", "landmark_nme_mean",
        "temporal_lpips_mean", "fid_score", "latency_p95_ms"
    ]
    
    # Write the header if file is missing or empty
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with open(output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        if write_header:
            writer.writeheader()
            
        for video_path in final_videos:
            # Reconstruct details
            parts = video_path.stem.split("_")
            vid_id = f"{parts[0]}_{parts[1]}_{parts[2]}" 
            target_speaker = parts[3]
            chunk_cond = set(parts) & {"C1", "C2", "C3", "C4", "C5"}
            chunk_condition = chunk_cond.pop() if chunk_cond else "unknown"
            precision_cond = set(parts) & {"fp32", "fp16", "int8"}
            precision = precision_cond.pop() if precision_cond else "unknown"
            architecture = parts[-1]
            
            source_video = videos.get(vid_id, {}).get("source_video")
            target_ref = Path("src/assets/target_image/elunma.jpeg").absolute() # Dynamically sourced target mapped via dev_mode
            
            # Temporary metrics output for this file
            tmp_json = video_path.parent / f"{video_path.stem}_metrics.json"
            
            print(f"  [EVAL] Running {architecture} metrics for {vid_id} ...")
            
            if video_metrics_script.exists():
                cmd = [
                    sys.executable, str(video_metrics_script),
                    "--generated", str(video_path.absolute()),
                    "--reference", str(target_ref),
                    "--source", str(Path(source_video).absolute()),
                    "--out", str(tmp_json.absolute())
                ]
                
                cpu_env = os.environ.copy()
                cpu_env["CUDA_VISIBLE_DEVICES"] = "-1"
                
                try:
                    # Execute interactively so user can see what metrics is failing on
                    print(" ".join(cmd))
                    subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, env=cpu_env, check=True)
                    
                    # Assume JSON gets populated, we read it
                    if tmp_json.exists():
                        with open(tmp_json, "r") as json_f:
                            data = json.load(json_f).get("metrics", {})
                            
                        # Example CSV formulation based on simulated/expected output
                        row = {
                            "vid_id": vid_id,
                            "target_speaker": target_speaker,
                            "chunk_condition": chunk_condition,
                            "precision": precision,
                            "architecture": architecture,
                            "identity_arcface_mean": data.get("identity_arcface", {}).get("mean"),
                            "landmark_nme_mean": data.get("landmark_nme", {}).get("mean_nme"),
                            "temporal_lpips_mean": data.get("temporal_lpips", {}).get("mean"),
                            "fid_score": data.get("fid", {}).get("fid"),
                            "latency_p95_ms": data.get("latency", {}).get("p95_latency_ms")
                        }
                        
                        writer.writerow(row)
                        
                except Exception as e:
                    print(f"  [ERROR] Video Evaluation failed for {video_path.name}: {e}")
                    
            else:
                 print(f"  [ERROR] Cannot find metrics script at {video_metrics_script}")

    print("\nPhase 5 Complete!")
    print(f"Master evaluation CSV updated at: {output_csv}")

if __name__ == "__main__":
    main()
