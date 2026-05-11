import argparse
import json
<<<<<<< HEAD
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
    
=======
import subprocess
import sys
from pathlib import Path

import pandas as pd


def extract_audio(video_path: Path, output_path: Path, target_sr: int = 16000) -> bool:
    """
    Extract mono 16 kHz PCM audio from a video file using ffmpeg directly.

    Why not librosa: PySoundFile (librosa's default backend) cannot decode
    mp4 containers — it only handles audio-only formats (wav, flac, ogg).
    librosa's audioread fallback works but is deprecated since 0.10 and
    will be removed in 1.0. ffmpeg handles every container we encounter,
    produces no warnings, and is already a hard dependency of the pipeline.
    """
    print(f"  Extracting: {video_path.name} -> {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", str(target_sr),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except FileNotFoundError:
        print("  [ERROR] ffmpeg not found in PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] ffmpeg failed (exit {e.returncode}) for {video_path.name}")
        return False


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load the FakeAVCeleb meta_data.csv.

    The official CSV has 9 header columns but 10 data columns per row.
    The extra column at index 8 contains the video filename (e.g. 00109.mp4).
    We detect both variants and assign canonical column names.
    """
    df = pd.read_csv(metadata_path)
    n = len(df.columns)

    if n == 10:
        df.columns = [
            "source", "target1", "target2", "method", "category",
            "type", "race", "gender", "filename", "path",
        ]
    elif n == 9:
        # Older format: no standalone filename column
        df.columns = [
            "source", "target1", "target2", "method", "category",
            "type", "race", "gender", "path",
        ]
        df["filename"] = None
    else:
        raise ValueError(f"Unexpected column count in metadata CSV: {n}")

    return df


def resolve_video_path(row: pd.Series, base_dir: Path) -> Path | None:
    """
    Resolve the actual mp4 path for a metadata row.

    Preference order:
      1. Exact filename from the CSV 'filename' column (most precise).
      2. First mp4 found in the expected folder, sorted alphabetically (fallback).
    """
    race = str(row.get("race", "")).strip()
    gender = str(row.get("gender", "")).strip()
    source_id = str(row.get("source", "")).strip()

    folder = base_dir / "RealVideo-RealAudio" / race / gender / source_id
    if not folder.is_dir():
        print(f"  [WARN] Folder not found: {folder}")
        return None

    filename_hint = str(row.get("filename", "") or "").strip()
    if filename_hint.endswith(".mp4"):
        candidate = folder / filename_hint
        if candidate.exists():
            return candidate
        print(f"  [WARN] CSV-specified file '{filename_hint}' not in {folder}, scanning...")

    mp4_files = sorted(folder.glob("*.mp4"))
    if not mp4_files:
        print(f"  [WARN] No .mp4 files in {folder}")
        return None

    return mp4_files[0]


def sample_with_diversity(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Stratified sampling across (race, gender) groups for maximum diversity.

    Strategy:
      1. Shuffle each group with the given seed.
      2. Round-robin one video per group until n is reached.
      3. If more slots remain (n > number of groups), fill by continuing
         round-robin through groups for a second pass, and so on.

    This guarantees that as long as n <= number of groups, every group gets
    exactly one representative.  For n=10 against FakeAVCeleb's ~12 groups
    (6 races × 2 genders) this produces near-maximum demographic variety.
    """
    rng = df.groupby(["race", "gender"], group_keys=False)

    # Shuffle each group independently with the same seed for reproducibility
    shuffled_groups = [
        group.sample(frac=1, random_state=seed)
        for _, group in rng
    ]

    # Round-robin pick: one from each group per round
    selected = []
    round_idx = 0  # how many we've already taken from each group this pass
    while len(selected) < n:
        added_this_round = 0
        for group in shuffled_groups:
            if len(selected) >= n:
                break
            if round_idx < len(group):
                selected.append(group.iloc[round_idx])
                added_this_round += 1
        if added_this_round == 0:
            # All groups exhausted — shouldn't happen with FakeAVCeleb + n<=500
            print(f"[WARN] Only {len(selected)} diverse samples available, requested {n}.")
            break
        round_idx += 1

    result = pd.DataFrame(selected)
    # Print diversity summary so the user can verify
    print(f"Diversity sample ({len(result)} videos):")
    summary = result.groupby(["race", "gender"]).size().reset_index(name="count")
    for _, row in summary.iterrows():
        print(f"  {row['race']:<25} {row['gender']:<6} -> {row['count']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Data Preparation")
    parser.add_argument("--base_dir", type=str, default="src/data/raw")
    parser.add_argument("--metadata_csv", type=str, default="data/FakeAVCeleb_v1.2/meta_data.csv")
    parser.add_argument("--output_audio_dir", type=str, default="data/processed_audio")
    parser.add_argument("--manifest_path", type=str, default="data/dataset_manifest.json")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
    base_dir = Path(args.base_dir)
    metadata_path = Path(args.metadata_csv)
    audio_out_dir = Path(args.output_audio_dir)
    manifest_path = Path(args.manifest_path)
<<<<<<< HEAD
    
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
=======

    audio_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        print(f"[ERROR] Metadata CSV not found: {metadata_path}")
        sys.exit(1)

    print(f"Loading metadata from {metadata_path}...")
    try:
        df = load_metadata(metadata_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse metadata CSV: {e}")
        sys.exit(1)

    real_subset = df[df["type"] == "RealVideo-RealAudio"].copy()
    print(f"Found {len(real_subset)} RealVideo-RealAudio entries.")

    if real_subset.empty:
        print("[ERROR] No real video entries found in metadata.")
        sys.exit(1)

    sampled_df = sample_with_diversity(real_subset, args.num_samples, args.seed)

    manifest = {
        "dataset_name": "FakeAVCeleb_Subset",
        "seed": args.seed,
        "videos": {},
    }

    success_count = 0
    print(f"\n--- Extracting audio for {len(sampled_df)} sampled videos ---")

    for _, row in sampled_df.iterrows():
        video_path = resolve_video_path(row, base_dir)
        if video_path is None:
            continue

        source_id = str(row.get("source", "")).strip()
        vid_id = f"video_{source_id}_{video_path.stem}"
        output_audio_path = audio_out_dir / f"{vid_id}.wav"

        # Skip if already extracted — allows partial re-runs without re-processing
        if output_audio_path.exists() and output_audio_path.stat().st_size > 0:
            print(f"  [SKIP] Already exists: {output_audio_path.name}")
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
            success_count += 1
            manifest["videos"][vid_id] = {
                "source_video": str(video_path.absolute()),
                "source_audio": str(output_audio_path.absolute()),
<<<<<<< HEAD
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
=======
                "race": str(row.get("race", "")).strip(),
                "gender": str(row.get("gender", "")).strip(),
            }
            continue

        if extract_audio(video_path, output_audio_path):
            success_count += 1
            manifest["videos"][vid_id] = {
                "source_video": str(video_path.absolute()),
                "source_audio": str(output_audio_path.absolute()),
                "race": str(row.get("race", "")).strip(),
                "gender": str(row.get("gender", "")).strip(),
            }
        else:
            print(f"  [WARN] Skipping {vid_id} — audio extraction failed.")

    # sampled_size reflects actual successful entries, not the sample request size
    manifest["sampled_size"] = success_count

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(f"\nPhase 1 complete.")
    print(f"Extracted {success_count}/{len(sampled_df)} tracks -> {audio_out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
