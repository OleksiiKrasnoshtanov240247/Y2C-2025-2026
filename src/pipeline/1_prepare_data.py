import argparse
import json
import os
import random
import subprocess
import sys
import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf

def extract_audio(video_path: Path, output_audio_path: Path, target_sr: int = 16000) -> bool:
    """Extract audio from video safely using librosa with ffmpeg fallback."""
    print(f"Extracting {video_path.name} -> {output_audio_path.name}")
    try:
        audio, sr = librosa.load(str(video_path), sr=target_sr, mono=True)
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_audio_path), audio, sr, format='WAV', subtype='PCM_16')
        return True
    except Exception as e:
        print(f"  [WARN] Librosa extraction failed: {e}. Trying fallback...")
        try:
            output_audio_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "ffmpeg", "-i", str(video_path), 
                "-ar", str(target_sr), "-ac", "1", 
                "-y", str(output_audio_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception as fallback_e:
            print(f"  [ERROR] Fallback failed: {fallback_e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Deepfake Data Preparation (Subset sampling)")
    parser.add_argument("--base_dir", type=str, default="src/data/raw", help="Base directory of FakeAVCeleb dataset")
    parser.add_argument("--metadata_csv", type=str, default="data/FakeAVCeleb_v1.2/meta_data.csv", help="Path to FakeAVCeleb metadata CSV")
    parser.add_argument("--output_audio_dir", type=str, default="data/processed_audio", help="Directory to save extracted 16kHz WAV files")
    parser.add_argument("--manifest_path", type=str, default="data/dataset_manifest.json", help="Path to save the generated JSON manifest")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of real videos to randomly sample for the run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    metadata_path = Path(args.metadata_csv)
    audio_out_dir = Path(args.output_audio_dir)
    manifest_path = Path(args.manifest_path)
    
    # Create necessary directories
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not metadata_path.exists():
        print(f"[ERROR] Metadata CSV not found at {metadata_path}")
        sys.exit(1)
        
    print(f"Loading metadata from {metadata_path}...")
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        sys.exit(1)
        
    # Filter for RealVideo-RealAudio (Category A)
    real_subset = df[df['type'] == 'RealVideo-RealAudio'].copy()
    print(f"Found {len(real_subset)} 'True Video / True Audio' entries.")
    
    if len(real_subset) == 0:
        print("[ERROR] No real videos found in metadata.")
        sys.exit(1)
        
    # Randomly sample --num_samples
    sampled_df = real_subset.sample(n=min(args.num_samples, len(real_subset)), random_state=args.seed)
    
    manifest = {
        "dataset_name": "FakeAVCeleb_Subset",
        "sampled_size": len(sampled_df),
        "seed": args.seed,
        "videos": {}
    }
    
    success_count = 0
    print(f"\n--- Extracting Audio for {len(sampled_df)} sampled videos ---")
    
    for idx, row in sampled_df.iterrows():
        # Clean path: the CSV 'path' looks like "FakeAVCeleb/RealVideo-RealAudio/African/men/id00076"
        # We need to map it to our actual disk layout
        raw_path = str(row['path'])
        filename = str(row['source']) # Note: source column doesn't have .mp4, but the 9th column 'path' string is weird in the user snippet, let's look at index 8 for path. Wait, the 9th column (index 8/9 depending on header) is 'path' and 'gender'
        # Actually in our View file: column 8 (index 8) is path: FakeAVCeleb/RealVideo-RealAudio/... column 0 is source: id00076, column 8 is path
        # And the full file path is path + filename ? Wait, let's just find the file.
        # the CSV has columns: source,target1,target2,method,category,type,race,gender,path
        # BUT line 2 is: id00076,-,-,real,A,RealVideo-RealAudio,African,men,00109.mp4,FakeAVCeleb/RealVideo-RealAudio/African/men/id00076
        # That's 10 values! source, target1, target2, method, category, type, race, gender, unnamed_col, path
        # Let's handle it safely by parsing the row dynamically or just hunting the directory
        
        # Safe construction
        race = str(row.get('race', 'African')).strip()
        gender = str(row.get('gender', 'men')).strip()
        source_id = str(row.get('source', 'id00000')).strip()
        
        # We just find the mp4 dynamically in the base_dir / RealVideo-RealAudio / race / gender / source_id
        target_folder = base_dir / "RealVideo-RealAudio" / race / gender / source_id
        
        if not target_folder.exists() or not target_folder.is_dir():
            print(f"  [WARN] Expected folder not found: {target_folder}")
            continue
            
        mp4_files = list(target_folder.glob("*.mp4"))
        if not mp4_files:
            print(f"  [WARN] No mp4 in {target_folder}")
            continue
            
        video_path = mp4_files[0] # Just take the first real video in that folder
        vid_id = f"video_{source_id}_{video_path.stem}"
        
        audio_name = f"{vid_id}.wav"
        output_audio_path = audio_out_dir / audio_name
        
        success = extract_audio(video_path, output_audio_path)
        
        if success:
            success_count += 1
            manifest["videos"][vid_id] = {
                "source_video": str(video_path.absolute()),
                "source_audio": str(output_audio_path.absolute()),
                "race": race,
                "gender": gender
            }
        else:
            print(f"  [WARN] Skipping {vid_id} due to extraction errors.")
            
    # Save the manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\nPhase 1 Complete!")
    print(f"Extracted {success_count}/{len(sampled_df)} audio tracks to {audio_out_dir}")
    print(f"Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()
