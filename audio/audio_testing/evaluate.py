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
    target_path: str | None = None,
    ground_truth_transcript: str | None = None,
    whisper_model: str = "large-v3-turbo",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    metrics: list[str] | None = None,
    inference_stats_path: str | None = None,
) -> dict:
    """Evaluate voice conversion quality.

    Args:
        original_path: Path to original (source) audio.
        transformed_path: Path to transformed (converted) audio.
        target_path: Path to target speaker reference audio. When provided,
            SECS compares converted vs target (correct for VC research).
            When omitted, SECS compares original vs converted (legacy).
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

    _W = 60  # output width for section separators

    print("=" * _W)
    print("  Audio Evaluation Pipeline")
    print("=" * _W)
    print(f"  Original    : {Path(original_path).name}")
    print(f"  Transformed : {Path(transformed_path).name}")
    if target_path:
        print(f"  Target      : {Path(target_path).name}")
    print("-" * _W)
    print("  Loading and preprocessing audio...")

    # Load with smart preprocessing (analyzes metadata, normalizes, resamples)
    original_processed, sr_orig, orig_metadata = prepare_audio_for_metrics(original_path)
    transformed_processed, sr_trans, trans_metadata = prepare_audio_for_metrics(transformed_path)

    # Load target speaker reference audio for SECS (if provided)
    target_processed = None
    if target_path is not None:
        target_processed, _, target_metadata = prepare_audio_for_metrics(target_path)

    results = {
        "original_file": str(Path(original_path).name),
        "transformed_file": str(Path(transformed_path).name),
        "target_file": str(Path(target_path).name) if target_path else None,
        "original_sample_rate": orig_metadata["original_sr"],
        "transformed_sample_rate": trans_metadata["original_sr"],
        "processing_sample_rate": STANDARD_SR,
        "original_duration_sec": round(len(original_processed) / STANDARD_SR, 3),
        "transformed_duration_sec": round(len(transformed_processed) / STANDARD_SR, 3),
    }

    # Load inference performance stats (from sidecar .perf.json if available)
    _stats_path = inference_stats_path
    if _stats_path is None:
        _auto = Path(transformed_path).with_suffix("").as_posix() + ".perf.json"
        if Path(_auto).is_file():
            _stats_path = _auto
    if _stats_path is not None and Path(_stats_path).is_file():
        with open(_stats_path) as _pf:
            _perf = json.load(_pf)
        results["perf_num_chunks"] = _perf.get("num_chunks")
        results["perf_mean_chunk_latency_ms"] = _perf.get("mean_chunk_latency_ms")
        results["perf_min_chunk_latency_ms"] = _perf.get("min_chunk_latency_ms")
        results["perf_max_chunk_latency_ms"] = _perf.get("max_chunk_latency_ms")
        results["perf_peak_gpu_memory_mb"] = _perf.get("peak_gpu_memory_mb")
        print(f"  Inference stats : {Path(_stats_path).name}")
    print("-" * _W)

    # Category 1: Content Preservation
    if "content" in metrics:
        print(f"\n  [1/4] Content Preservation")
        print(f"  Whisper model : {whisper_model}")
        print(f"  {'─' * 40}")
        t0 = time.time()

        if ground_truth_transcript is None:
            print("  Transcribing original audio as ground truth...")
            from metrics.intelligibility import transcribe_audio_whisper
            ground_truth_transcript = transcribe_audio_whisper(
                original_processed, STANDARD_SR,
                model_name=whisper_model,
                device=whisper_device,
                compute_type=whisper_compute_type
            )
            preview = ground_truth_transcript[:80]
            print(f"  Ground truth  : \"{preview}{'...' if len(ground_truth_transcript) > 80 else ''}\"")

        print("  Transcribing transformed audio...")
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

        print(f"  WER           : {results['content_wer']:.4f}  (<0.10 excellent, <0.20 good)")
        print(f"  CER           : {results['content_cer']:.4f}  (<0.05 excellent, <0.15 good)")
        print(f"  Time          : {results['content_time_sec']}s")

    # Category 2: Perceptual Quality
    if "quality" in metrics:
        print(f"\n  [2/4] Perceptual Quality")
        print(f"  {'─' * 40}")

        print("  Computing STOI...")
        t0 = time.time()
        stoi_score = compute_stoi(original_processed, transformed_processed, STANDARD_SR)
        results["quality_stoi"] = round(stoi_score, 4)
        results["stoi_time_sec"] = round(time.time() - t0, 2)
        print(f"  STOI          : {results['quality_stoi']:.4f}  (>0.85 excellent, >0.70 good)")

        print("  Computing PESQ...")
        t0 = time.time()
        try:
            pesq_score = compute_pesq(original_processed, transformed_processed, STANDARD_SR)
            results["quality_pesq"] = round(pesq_score, 4)
            results["pesq_time_sec"] = round(time.time() - t0, 2)
            print(f"  PESQ          : {results['quality_pesq']:.2f}   (>3.5 excellent, >2.5 good)")
        except Exception as e:
            print(f"  PESQ          : failed — {e}")
            results["quality_pesq"] = None

        print("  Computing SNR...")
        t0 = time.time()
        snr = compute_snr(transformed_processed)
        results["quality_snr_db"] = round(snr, 2)
        results["snr_time_sec"] = round(time.time() - t0, 2)
        print(f"  SNR           : {results['quality_snr_db']:.2f} dB  (>30 dB excellent, >20 dB good)")

    # Category 3: Voice Conversion Quality
    if "voice_change" in metrics:
        print(f"\n  [3/4] Voice Conversion Quality")
        print(f"  {'─' * 40}")

        print("  Computing MCD...")
        t0 = time.time()
        mcd_score = compute_mcd(original_processed, transformed_processed, STANDARD_SR)
        results["conversion_mcd_db"] = round(mcd_score, 4)
        results["mcd_time_sec"] = round(time.time() - t0, 2)
        print(f"  MCD           : {results['conversion_mcd_db']:.2f} dB  (<5.0 excellent, <7.0 good)")

        print("  Computing SECS...")
        t0 = time.time()
        if target_processed is not None:
            # Compare converted vs target speaker (correct for VC research)
            secs_score = compute_secs(transformed_processed, target_processed, STANDARD_SR)
            results["secs_comparison"] = "converted_vs_target"
        else:
            # Fallback: compare original vs converted (no target provided)
            secs_score = compute_secs(original_processed, transformed_processed, STANDARD_SR)
            results["secs_comparison"] = "original_vs_converted"
        results["conversion_secs"] = round(secs_score, 4)
        results["secs_time_sec"] = round(time.time() - t0, 2)
        if target_processed is not None:
            print(f"  SECS          : {results['conversion_secs']:.4f}  (higher = closer to target speaker)")
        else:
            print(f"  SECS          : {results['conversion_secs']:.4f}  (lower = more voice change)")

    # Category 4: Prosody Preservation
    if "prosody" in metrics:
        print(f"\n  [4/4] Prosody Preservation")
        print(f"  {'─' * 40}")

        print("  Computing F0 metrics...")
        t0 = time.time()
        f0_results = compute_f0_metrics(original_processed, transformed_processed, STANDARD_SR)
        results["prosody_f0_correlation"] = round(f0_results["f0_correlation"], 4)
        results["prosody_f0_rmse_hz"] = round(f0_results["f0_rmse"], 4)
        results["prosody_voiced_frames"] = f0_results["voiced_frames"]
        results["f0_time_sec"] = round(time.time() - t0, 2)
        print(f"  F0 Correlation: {results['prosody_f0_correlation']:.4f}  (>0.80 excellent, >0.60 good)")
        print(f"  F0 RMSE       : {results['prosody_f0_rmse_hz']:.2f} Hz  (<20 Hz excellent, <40 Hz good)")

    return results


def _rating(value: float, thresholds: list[tuple[float, str]], reverse: bool = False) -> str:
    """Return a rating label based on ordered thresholds.

    Args:
        value: Metric value.
        thresholds: List of (threshold, label) pairs, ordered best-to-worst.
        reverse: If True, lower values are better (e.g. WER, MCD).
    """
    for threshold, label in thresholds:
        if (not reverse and value >= threshold) or (reverse and value <= threshold):
            return label
    return thresholds[-1][1]


def print_summary(results: dict) -> None:
    """Print formatted evaluation summary."""
    W = 70
    print("\n" + "=" * W)
    print("  EVALUATION SUMMARY")
    print("=" * W)
    print(f"  Original    : {results['original_file']}  ({results['original_sample_rate']} Hz)")
    print(f"  Transformed : {results['transformed_file']}  ({results['transformed_sample_rate']} Hz)")
    print(f"  Duration    : {results['original_duration_sec']}s  ->  {results['transformed_duration_sec']}s")
    print("-" * W)

    # Content Preservation
    if "content_wer" in results:
        print("\n  Content Preservation")
        wer = results["content_wer"]
        cer = results["content_cer"]
        wer_rating = _rating(wer, [(0.10, "Excellent"), (0.20, "Good"), (0.35, "Fair")], reverse=True)
        print(f"  {'WER (Word Error Rate)':<30} {wer:.4f}   [{wer_rating}]")
        print(f"  {'CER (Character Error Rate)':<30} {cer:.4f}")

    # Perceptual Quality
    if "quality_stoi" in results:
        print("\n  Perceptual Quality")
        stoi = results["quality_stoi"]
        pesq = results.get("quality_pesq")
        snr  = results.get("quality_snr_db")

        stoi_rating = _rating(stoi, [(0.85, "Excellent"), (0.70, "Good"), (0.50, "Fair")])
        print(f"  {'STOI (Intelligibility)':<30} {stoi:.4f}   [{stoi_rating}]")

        if pesq is not None:
            pesq_rating = _rating(pesq, [(3.5, "Excellent"), (2.5, "Good"), (1.5, "Fair")])
            print(f"  {'PESQ (Speech Quality)':<30} {pesq:.2f}     [{pesq_rating}]")

        if snr is not None:
            snr_rating = _rating(snr, [(30.0, "Excellent"), (20.0, "Good"), (0.0, "Fair")])
            print(f"  {'SNR (Signal-to-Noise)':<30} {snr:.2f} dB  [{snr_rating}]")

    # Voice Conversion Quality
    if "conversion_mcd_db" in results:
        print("\n  Voice Conversion Quality")
        mcd  = results["conversion_mcd_db"]
        secs = results.get("conversion_secs")

        mcd_rating = _rating(mcd, [(5.0, "Excellent"), (7.0, "Good"), (10.0, "Fair")], reverse=True)
        print(f"  {'MCD (Mel Cepstral Dist.)':<30} {mcd:.2f} dB  [{mcd_rating}]")

        if secs is not None:
            comparison = results.get("secs_comparison", "original_vs_converted")
            if comparison == "converted_vs_target":
                if secs > 0.85:
                    secs_note = "excellent target match"
                elif secs > 0.70:
                    secs_note = "good target match"
                else:
                    secs_note = "weak target match"
                print(f"  {'SECS (vs Target Speaker)':<30} {secs:.4f}   [{secs_note}]")
            else:
                if secs < 0.30:
                    secs_note = "significant voice change"
                elif secs < 0.60:
                    secs_note = "moderate voice change"
                else:
                    secs_note = "minimal voice change"
                print(f"  {'SECS (Speaker Similarity)':<30} {secs:.4f}   [{secs_note}]")

    # Prosody Preservation
    if "prosody_f0_correlation" in results:
        print("\n  Prosody Preservation")
        f0_corr = results["prosody_f0_correlation"]
        f0_rmse = results["prosody_f0_rmse_hz"]
        f0_rating = _rating(f0_corr, [(0.80, "Excellent"), (0.60, "Good"), (0.40, "Fair")])
        print(f"  {'F0 Correlation':<30} {f0_corr:.4f}   [{f0_rating}]")
        rmse_rating = _rating(f0_rmse, [(20.0, "Excellent"), (40.0, "Good"), (999.0, "Fair")], reverse=True)
        print(f"  {'F0 RMSE':<30} {f0_rmse:.2f} Hz  [{rmse_rating}]")

    # Inference Performance
    if "perf_mean_chunk_latency_ms" in results:
        print("\n  Inference Performance")
        n       = results.get("perf_num_chunks", "N/A")
        mean_ms = results.get("perf_mean_chunk_latency_ms")
        min_ms  = results.get("perf_min_chunk_latency_ms")
        max_ms  = results.get("perf_max_chunk_latency_ms")
        gpu_mb  = results.get("perf_peak_gpu_memory_mb")

        print(f"  {'Chunks processed':<30} {n}")
        if mean_ms is not None:
            print(f"  {'Chunk latency (mean)':<30} {mean_ms:.1f} ms")
        if min_ms is not None and max_ms is not None:
            print(f"  {'Chunk latency (min / max)':<30} {min_ms:.1f} ms  /  {max_ms:.1f} ms")
        if gpu_mb is not None:
            print(f"  {'Peak GPU memory':<30} {gpu_mb:.1f} MB")
        else:
            print(f"  {'Peak GPU memory':<30} N/A (CPU inference)")

    print("\n" + "=" * W)


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
  python evaluate.py --original input.wav --transformed output.wav --target target_speaker.wav
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
        "--target", type=str, default=None,
        help="Path to target speaker reference audio (for SECS: compares converted vs target)",
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
        "--inference-stats", type=str, default=None, dest="inference_stats",
        help="Path to .perf.json sidecar produced by Applio infer with measure_performance=True "
             "(auto-detected if <transformed>.perf.json exists alongside the audio file)",
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
    if args.target and not Path(args.target).is_file():
        print(f"Error: target file not found: {args.target}", file=sys.stderr)
        sys.exit(1)

    results = evaluate(
        original_path=args.original,
        transformed_path=args.transformed,
        target_path=args.target,
        ground_truth_transcript=args.transcript,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute_type,
        metrics=args.metrics,
        inference_stats_path=args.inference_stats,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
