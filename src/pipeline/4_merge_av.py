import argparse
<<<<<<< HEAD
import os
=======
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
import subprocess
import sys
from pathlib import Path

<<<<<<< HEAD
def merge_audio_video(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Uses FFMPEG to multiplex the audio into the video, strictly replacing existing audio."""
    print(f"Multiplexing:\n  [V] {video_path.name}\n  [A] {audio_path.name}\n  -> {output_path.name}")
    
    # -i video -i audio -c:v copy (copy video) -c:a aac (encode audio) -map 0:v:0 (video from input 0) -map 1:a:0 (audio from input 1)
    cmd = [
        "ffmpeg", "-y", 
        "-i", str(video_path), 
        "-i", str(audio_path), 
=======

def merge_audio_video(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """
    Multiplex audio into video using ffmpeg.

    -c:v copy   — video stream is remuxed without re-encoding (fast, lossless)
    -c:a aac    — audio is encoded to AAC (required for broad mp4 compatibility)
    -shortest   — output duration matches the shorter of the two streams
    """
    print(f"Multiplexing:\n  [V] {video_path.name}\n  [A] {audio_path.name}\n  -> {output_path.name}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
<<<<<<< HEAD
        "-shortest", # Finish encoding when the shortest input stream ends
        str(output_path)
    ]
    
    try:
        # Hide output unless error occurs
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] FFMPEG Failed: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        print("  [ERROR] FFMPEG is not installed or not in PATH.")
        return False
    except Exception as e:
        print(f"  [ERROR] Multiplexing failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Phase 4: AV Merging Pipeline")
    parser.add_argument("--video_dir", type=str, default="data/generated_video", help="Directory with generated video files")
    parser.add_argument("--audio_dir", type=str, default="data/generated_audio", help="Directory with generated audio files")
    parser.add_argument("--output_dir", type=str, default="data/final_merged", help="Directory to save final merged AV files")
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_dir.exists() or not audio_dir.exists():
        print(f"[ERROR] Required directories do not exist. Please run Phase 2 and 3 first.")
        sys.exit(1)
        
    videos = list(video_dir.glob("*.mp4"))
    
    print(f"\n--- Starting AV Merging ---")
    print(f"Found {len(videos)} videos to merge.")
    
    success_count = 0
    
    for video_path in videos:
        # Expected naming from Phase 3:
        # <vid_id>_<target_speaker>_<chunk_condition>_<precision>_<architecture>.mp4
        
        # We need to find the specific audio file generated for this condition
        # Example video name: video_001_id123_DonaldTrump_C3_fp32_MuseTalk.mp4
        # Corresponding audio: video_001_id123_DonaldTrump_C3_fp32.wav
        
        # Strip the _<architecture>.mp4
        parts = video_path.stem.split("_")
        
        if len(parts) < 6:
            print(f"  [WARN] Skipping {video_path.name}, unusual naming convention.")
            continue
            
        architecture = parts[-1]
        base_audio_name = video_path.stem.replace(f"_{architecture}", "")
        
        audio_path = audio_dir / f"{base_audio_name}.wav"
        
        if not audio_path.exists():
            print(f"  [WARN] Skipping {video_path.name}, no corresponding audio {audio_path.name} found.")
            continue
            
        output_path = output_dir / video_path.name
        
        # Execute merge
        success = merge_audio_video(video_path, audio_path, output_path)
        if success:
            success_count += 1
            
    print(f"\nPhase 4 Complete!")
    print(f"Merged {success_count}/{len(videos)} videos.")
    print(f"Final stimuli saved to {output_dir}")

if __name__ == "__main__":
    main()
=======
        "-shortest",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] ffmpeg failed:\n{result.stderr.strip()}")
            return False
        return True
    except FileNotFoundError:
        print("  [ERROR] ffmpeg not found in PATH.")
        return False
    except Exception as e:
        print(f"  [ERROR] Unexpected error during mux: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Phase 4: AV Merging")
    parser.add_argument("--video_dir", type=str, default="data/generated_video")
    parser.add_argument("--audio_dir", type=str, default="data/generated_audio")
    parser.add_argument("--output_dir", type=str, default="data/final_merged")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        print(f"[ERROR] Video directory not found: {video_dir}")
        sys.exit(1)
    if not audio_dir.exists():
        print(f"[ERROR] Audio directory not found: {audio_dir}")
        sys.exit(1)

    videos = sorted(video_dir.glob("*.mp4"))
    print(f"\n--- Starting AV Merging ---")
    print(f"Found {len(videos)} videos.")

    success_count = 0
    skip_count = 0

    for video_path in videos:
        output_path = output_dir / video_path.name

        if output_path.exists() and output_path.stat().st_size > 0:
            skip_count += 1
            continue

        # Naming convention:
        #   video:  {vid_id}_{speaker}_{condition}_{precision}_{architecture}.mp4
        #   audio:  {vid_id}_{speaker}_{condition}_{precision}.wav
        #
        # rsplit("_", 1) strips only the last segment (architecture name).
        # str.replace() is avoided here because it matches the first occurrence
        # anywhere in the string, which would corrupt names if the architecture
        # string appeared elsewhere (e.g. a future architecture named "fp32").
        audio_base = video_path.stem.rsplit("_", 1)[0]
        audio_path = audio_dir / f"{audio_base}.wav"

        if not audio_path.exists():
            print(f"  [WARN] No matching audio for {video_path.name} (expected {audio_path.name})")
            continue

        if merge_audio_video(video_path, audio_path, output_path):
            success_count += 1

    print(f"\nPhase 4 complete.")
    print(f"Merged: {success_count}  |  Skipped (already done): {skip_count}  |  Total: {len(videos)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
