"""
Audio Evaluation Pipeline for Voice Conversion

Evaluates voice conversion quality by measuring content preservation,
perceptual quality, spectral conversion, and prosody preservation.

Usage:
    python evaluate.py --original input.wav --transformed output.wav
    python evaluate.py --original input.wav --transformed output.wav --transcript "text"
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from audio_utils import load_audio, prepare_audio_for_metrics, STANDARD_SR
from metrics.speaker_similarity import compute_secs, compute_mcd, compute_f0_metrics
from metrics.intelligibility import compute_wer_cer_single, compute_stoi
from metrics.quality import compute_pesq, compute_snr

ALL_METRICS = ["content", "quality", "voice_change", "prosody"]


def evaluate(
    original_path: str,
    transformed_path: str,
    ground_truth_transcript: str | None = None,
    whisper_model: str = "large-v3-turbo",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    metrics: list[str] | None = None,
) -> dict:
    """Evaluate voice conversion quality.

    Args:
        original_path: Path to original audio.
        transformed_path: Path to transformed audio.
        ground_truth_transcript: Optional ground truth transcript.
        whisper_model: Whisper model for transcription.
        whisper_device: Device for Whisper.
        whisper_compute_type: Quantization type.
        metrics: Metric categories to compute (default: all).

    Returns:
        Dict with evaluation results.
    """
    if metrics is None:
        metrics = ALL_METRICS

    print("Loading and preprocessing audio files...")
    
    # Load with smart preprocessing (analyzes metadata, normalizes, resamples)
    original_processed, sr_orig, orig_metadata = prepare_audio_for_metrics(original_path)
    transformed_processed, sr_trans, trans_metadata = prepare_audio_for_metrics(transformed_path)

    results = {
        "original_file": str(Path(original_path).name),
        "transformed_file": str(Path(transformed_path).name),
        "original_sample_rate": orig_metadata["original_sr"],
        "transformed_sample_rate": trans_metadata["original_sr"],
        "processing_sample_rate": STANDARD_SR,
        "original_duration_sec": round(len(original_processed) / STANDARD_SR, 3),
        "transformed_duration_sec": round(len(transformed_processed) / STANDARD_SR, 3),
    }

    # Category 1: Content Preservation
    # Goal: Transformed audio should convey the same words/content as original
    # WER should be LOW (< 10% is excellent)
    
    if "content" in metrics:
        print("\nContent Preservation (WER/CER)")
        print(f"Computing WER & CER using Whisper {whisper_model}...")
        t0 = time.time()

        # Get ground truth transcript
        if ground_truth_transcript is None:
            print("  → Transcribing original audio as ground truth...")
            from metrics.intelligibility import transcribe_audio_whisper
            ground_truth_transcript = transcribe_audio_whisper(
                original_processed, STANDARD_SR,
                model_name=whisper_model,
                device=whisper_device,
                compute_type=whisper_compute_type
            )
            print(f"  → Ground truth: \"{ground_truth_transcript[:80]}...\"")
        
        # Transcribe transformed audio
        print("  → Transcribing transformed audio...")
        wer_result = compute_wer_cer_single(
            transformed_processed, STANDARD_SR,
            ground_truth_transcript=ground_truth_transcript,
            whisper_model_name=whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )
        
        results["content_wer"] = round(wer_result["wer"], 4)
        results["content_cer"] = round(wer_result["cer"], 4)
        results["ground_truth_transcript"] = wer_result["reference_transcript"]
        results["transformed_transcript"] = wer_result["hypothesis_transcript"]
        results["content_time_sec"] = round(time.time() - t0, 2)
        
        print(f"  WER: {results['content_wer']:.4f} (lower is better, <0.10 excellent, <0.20 good)")
        print(f"  CER: {results['content_cer']:.4f} (lower is better, <0.05 excellent, <0.15 good)")

    # Category 2: Perceptual Quality
    # Goal: Transformed audio should sound natural and high-quality
    
    if "quality" in metrics:
        print("\nPerceptual Quality")
        
        print("Computing STOI (Short-Time Objective Intelligibility)...")
        t0 = time.time()
        stoi_score = compute_stoi(original_processed, transformed_processed, STANDARD_SR)
        results["quality_stoi"] = round(stoi_score, 4)
        results["stoi_time_sec"] = round(time.time() - t0, 2)
        print(f"  STOI: {results['quality_stoi']:.4f} (range 0-1, >0.85 excellent, >0.70 good)")
        
        print("Computing PESQ (Perceptual Evaluation of Speech Quality)...")
        t0 = time.time()
        try:
            pesq_score = compute_pesq(original_processed, transformed_processed, STANDARD_SR)
            results["quality_pesq"] = round(pesq_score, 4)
            results["pesq_time_sec"] = round(time.time() - t0, 2)
            print(f"  PESQ: {results['quality_pesq']:.2f} (range 1.0-4.5, >3.5 excellent, >2.5 good)")
        except Exception as e:
            print(f"  PESQ failed: {e}")
            results["quality_pesq"] = None
        
        print("Computing SNR (Signal-to-Noise Ratio estimate)...")
        t0 = time.time()
        snr = compute_snr(transformed_processed)
        results["quality_snr_db"] = round(snr, 2)
        results["snr_time_sec"] = round(time.time() - t0, 2)
        print(f"  SNR: {results['quality_snr_db']:.2f} dB (>30 dB excellent, >20 dB good)")

    # Category 3: Voice Conversion Quality
    # Goal: Smooth spectral conversion with natural formant transitions
    
    if "voice_change" in metrics:
        print("\nVoice Conversion Quality")
        
        print("Computing MCD (Mel Cepstral Distortion)...")
        t0 = time.time()
        mcd_score = compute_mcd(original_processed, transformed_processed, STANDARD_SR)
        results["conversion_mcd_db"] = round(mcd_score, 4)
        results["mcd_time_sec"] = round(time.time() - t0, 2)
        print(f"  MCD: {results['conversion_mcd_db']:.2f} dB (lower is better, <5.0 excellent, <7.0 good)")
        
        print("Computing SECS (Speaker Embedding Cosine Similarity)...")
        t0 = time.time()
        secs_score = compute_secs(original_processed, transformed_processed, STANDARD_SR)
        results["conversion_secs"] = round(secs_score, 4)
        results["secs_time_sec"] = round(time.time() - t0, 2)
        print(f"  SECS: {results['conversion_secs']:.4f} (range -1 to 1, lower means more voice change)")

    # Category 4: Prosody Preservation
    # Goal: Preserve pitch contours and rhythm
    
    if "prosody" in metrics:
        print("\nProsody Preservation")
        
        print("Computing F0 metrics (pitch correlation & RMSE)...")
        t0 = time.time()
        f0_results = compute_f0_metrics(original_processed, transformed_processed, STANDARD_SR)
        results["prosody_f0_correlation"] = round(f0_results["f0_correlation"], 4)
        results["prosody_f0_rmse_hz"] = round(f0_results["f0_rmse"], 4)
        results["prosody_voiced_frames"] = f0_results["voiced_frames"]
        results["f0_time_sec"] = round(time.time() - t0, 2)
        print(f"  F0 Correlation: {results['prosody_f0_correlation']:.4f} (range -1 to 1, >0.80 excellent, >0.60 good)")
        print(f"  F0 RMSE: {results['prosody_f0_rmse_hz']:.2f} Hz (lower is better, <20 Hz excellent, <40 Hz good)")

    return results


def print_summary(results: dict) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "-" * 70)
    print("Audio Evaluation Results")
    print("-" * 70)
    print(f"Original:    {results['original_file']} ({results['original_sample_rate']}Hz)")
    print(f"Transformed: {results['transformed_file']} ({results['transformed_sample_rate']}Hz)")
    print(f"Duration:    {results['original_duration_sec']}s to {results['transformed_duration_sec']}s")
    print("-" * 70)

    # Content Preservation
    if "content_wer" in results:
        print("\nContent Preservation")
        wer = results["content_wer"]
        cer = results["content_cer"]
        
        print(f"  WER (Word Error Rate):      {wer:.4f}")
        if wer < 0.10:
            print(f"    Assessment: Excellent content preservation")
        elif wer < 0.20:
            print(f"    Assessment: Good content preservation")
        elif wer < 0.35:
            print(f"    Assessment: Fair content preservation")
        else:
            print(f"    Assessment: Poor content preservation")
        
        print(f"  CER (Character Error Rate): {cer:.4f}")

    # Perceptual Quality
    if "quality_stoi" in results:
        print("\nPerceptual Quality")
        stoi = results["quality_stoi"]
        pesq = results.get("quality_pesq")
        snr = results.get("quality_snr_db")
        
        print(f"  STOI (Intelligibility):     {stoi:.4f} (range 0-1)")
        if stoi > 0.85:
            print(f"    Assessment: Excellent intelligibility")
        elif stoi > 0.70:
            print(f"    Assessment: Good intelligibility")
        elif stoi > 0.50:
            print(f"    Assessment: Fair intelligibility")
        else:
            print(f"    Assessment: Poor intelligibility")
        
        if pesq is not None:
            print(f"  PESQ (Speech Quality):      {pesq:.2f} (range 1.0-4.5)")
            if pesq > 3.5:
                print(f"    Assessment: Excellent speech quality")
            elif pesq > 2.5:
                print(f"    Assessment: Good speech quality")
            elif pesq > 1.5:
                print(f"    Assessment: Fair speech quality")
            else:
                print(f"    Assessment: Poor speech quality")
        
        if snr is not None:
            print(f"  SNR (Signal-to-Noise):      {snr:.2f} dB")
            if snr > 30:
                print(f"    Assessment: Excellent SNR")
            elif snr > 20:
                print(f"    Assessment: Good SNR")
            else:
                print(f"    Assessment: Fair SNR")

    # Voice Conversion Quality
    if "conversion_mcd_db" in results:
        print("\nVoice Conversion Quality")
        mcd = results["conversion_mcd_db"]
        secs = results.get("conversion_secs")
        
        print(f"  MCD (Mel Cepstral Dist):    {mcd:.2f} dB (lower is better)")
        if mcd < 5.0:
            print(f"    Assessment: Excellent spectral similarity")
        elif mcd < 7.0:
            print(f"    Assessment: Good spectral similarity")
        elif mcd < 10.0:
            print(f"    Assessment: Fair spectral similarity")
        else:
            print(f"    Assessment: Poor spectral similarity")
        
        if secs is not None:
            print(f"  SECS (Speaker Similarity):  {secs:.4f} (range -1 to 1)")
            print(f"    Note: Lower SECS indicates more voice change")
            if secs < 0.30:
                print(f"    Assessment: Significant voice change achieved")
            elif secs < 0.60:
                print(f"    Assessment: Moderate voice change")
            else:
                print(f"    Assessment: Minimal voice change")

    # Prosody Preservation
    if "prosody_f0_correlation" in results:
        print("\nProsody Preservation")
        f0_corr = results["prosody_f0_correlation"]
        f0_rmse = results["prosody_f0_rmse_hz"]
        
        print(f"  F0 Correlation:             {f0_corr:.4f} (range -1 to 1)")
        if f0_corr > 0.80:
            print(f"    Assessment: Excellent pitch preservation")
        elif f0_corr > 0.60:
            print(f"    Assessment: Good pitch preservation")
        elif f0_corr > 0.40:
            print(f"    Assessment: Fair pitch preservation")
        else:
            print(f"    Assessment: Poor pitch preservation")
        
        print(f"  F0 RMSE:                    {f0_rmse:.2f} Hz (lower is better)")

    print("-" * 70)


def save_results(results: dict, output_path: str) -> None:
    """Save results to CSV."""
    path = Path(output_path)
    file_exists = path.exists()

    csv_results = {k: v for k, v in results.items()
                   if k not in ("ground_truth_transcript", "transformed_transcript")}

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_results)

    print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Audio Evaluation Pipeline for Voice Conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metric Categories:
  content      Content preservation (WER, CER)
  quality      Perceptual quality (STOI, PESQ, SNR)
  voice_change Voice conversion quality (MCD, SECS)
  prosody      Prosody preservation (F0 correlation, RMSE)

Examples:
  python evaluate.py --original input.wav --transformed output.wav
  python evaluate.py --original input.wav --transformed output.wav --transcript "hello"
  python evaluate.py --original input.wav --transformed output.wav --metrics content quality
        """,
    )

    parser.add_argument(
        "--original", required=True,
        help="Path to original input audio file",
    )
    parser.add_argument(
        "--transformed", required=True,
        help="Path to transformed output audio file",
    )
    parser.add_argument(
        "--transcript", type=str, default=None,
        help="Ground truth transcript (optional; will auto-transcribe original if omitted)",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=ALL_METRICS, default=None,
        help="Metric categories to compute (default: all)",
    )
    parser.add_argument(
        "--whisper-model", type=str, default="large-v3-turbo",
        help="Whisper model for transcription (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--whisper-device", type=str, default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for Whisper (default: cpu)",
    )
    parser.add_argument(
        "--whisper-compute-type", type=str, default="int8",
        choices=["int8", "float16", "float32"],
        help="Quantization for faster-whisper (default: int8)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (optional)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print results as JSON",
    )

    args = parser.parse_args()

    # Validate files
    if not Path(args.original).is_file():
        print(f"Error: original file not found: {args.original}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.transformed).is_file():
        print(f"Error: transformed file not found: {args.transformed}", file=sys.stderr)
        sys.exit(1)

    results = evaluate(
        original_path=args.original,
        transformed_path=args.transformed,
        ground_truth_transcript=args.transcript,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute_type,
        metrics=args.metrics,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
