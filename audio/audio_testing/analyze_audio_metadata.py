"""
Audio metadata analyzer - collect detailed audio characteristics before processing.
Helps determine optimal preprocessing strategy.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json

from audio_utils import _load_raw_audio


def analyze_audio_file(path: str) -> dict:
    """Collect comprehensive audio metadata."""

    # Load raw audio without resampling (handles WAV, MP3, OGG, FLAC, etc.)
    audio_raw, sr_original = _load_raw_audio(path)

    # Convert stereo to mono if needed
    if len(audio_raw.shape) > 1:
        audio_raw = np.mean(audio_raw, axis=1)
    
    # Load with librosa for additional analysis
    audio_librosa, _ = librosa.load(path, sr=None, mono=True)
    
    # Basic properties
    duration = len(audio_raw) / sr_original
    
    # Dynamic range analysis
    rms = librosa.feature.rms(y=audio_librosa)[0]
    
    # Frequency content
    stft = np.abs(librosa.stft(audio_librosa))
    freqs = librosa.fft_frequencies(sr=sr_original)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_librosa, sr=sr_original)[0]
    
    # Pitch analysis
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_librosa,
        fmin=librosa.note_to_hz('C2'), # type: ignore
        fmax=librosa.note_to_hz('C7'), # type: ignore
        sr=sr_original
    )
    f0_clean = f0[~np.isnan(f0)]
    
    # Voice activity detection
    intervals = librosa.effects.split(audio_librosa, top_db=30)
    voiced_duration = sum([(end - start) / sr_original for start, end in intervals])
    
    # Signal-to-noise ratio estimate
    signal_power = np.mean(rms ** 2)
    noise_floor = np.percentile(rms, 10) ** 2
    snr_estimate = 10 * np.log10(signal_power / (noise_floor + 1e-10))
    
    # Clipping detection
    max_amplitude = np.max(np.abs(audio_raw))
    is_clipped = max_amplitude >= 0.99
    clipping_percentage = np.sum(np.abs(audio_raw) >= 0.99) / len(audio_raw) * 100
    
    metadata = {
        "file": str(Path(path).name),
        "format": {
            "sample_rate": int(sr_original),
            "duration_sec": round(duration, 3),
            "num_samples": len(audio_raw),
            "bit_depth": "float32" if audio_raw.dtype == np.float32 else str(audio_raw.dtype),
        },
        "amplitude": {
            "max": round(float(max_amplitude), 4),
            "rms_mean": round(float(np.mean(rms)), 4),
            "rms_std": round(float(np.std(rms)), 4),
            "dynamic_range_db": round(float(20 * np.log10(np.max(rms) / (np.min(rms[rms > 0]) + 1e-10))), 2),
            "clipped": bool(is_clipped),
            "clipping_pct": round(float(clipping_percentage), 4),
        },
        "frequency": {
            "spectral_centroid_mean_hz": round(float(np.mean(spectral_centroid)), 2),
            "spectral_centroid_std_hz": round(float(np.std(spectral_centroid)), 2),
            "nyquist_freq_hz": int(sr_original // 2),
        },
        "pitch": {
            "f0_mean_hz": round(float(np.mean(f0_clean)), 2) if len(f0_clean) > 0 else None,
            "f0_std_hz": round(float(np.std(f0_clean)), 2) if len(f0_clean) > 0 else None,
            "f0_min_hz": round(float(np.min(f0_clean)), 2) if len(f0_clean) > 0 else None,
            "f0_max_hz": round(float(np.max(f0_clean)), 2) if len(f0_clean) > 0 else None,
            "voiced_frames": int(np.sum(voiced_flag)),
            "voiced_percentage": round(float(np.sum(voiced_flag) / len(voiced_flag) * 100), 2),
        },
        "quality": {
            "snr_estimate_db": round(float(snr_estimate), 2),
            "voice_activity_duration_sec": round(float(voiced_duration), 3),
            "silence_percentage": round(float((1 - voiced_duration / duration) * 100), 2),
        },
        "recommendations": [],
    }
    
    # Generate recommendations
    if sr_original < 16000:
        metadata["recommendations"].append(f"WARNING: Sample rate {sr_original}Hz is below 16kHz - may limit quality")
    elif sr_original > 48000:
        metadata["recommendations"].append(f"INFO: High sample rate {sr_original}Hz - can safely downsample to 16-24kHz")
    
    if is_clipped:
        metadata["recommendations"].append(f"WARNING: Audio is clipped ({clipping_percentage:.2f}%) - may affect metrics")
    
    if snr_estimate < 20:
        metadata["recommendations"].append(f"WARNING: Low SNR ({snr_estimate:.1f}dB) - noisy recording")
    elif snr_estimate > 35:
        metadata["recommendations"].append(f"INFO: Clean recording (SNR {snr_estimate:.1f}dB)")
    
    if voiced_duration / duration < 0.5:
        metadata["recommendations"].append(f"WARNING: Low voice activity ({voiced_duration/duration*100:.1f}%) - mostly silence")
    
    return metadata


def compare_audio_pair(original_path: str, transformed_path: str):
    """Compare metadata of original vs transformed audio."""
    
    print("Analyzing audio pair metadata...\n")
    
    original_meta = analyze_audio_file(original_path)
    transformed_meta = analyze_audio_file(transformed_path)
    
    print("-" * 80)
    print("ORIGINAL AUDIO")
    print("-" * 80)
    print_metadata(original_meta)
    
    print("\n" + "-" * 80)
    print("TRANSFORMED AUDIO")
    print("-" * 80)
    print_metadata(transformed_meta)
    
    print("\n" + "-" * 80)
    print("COMPARISON & RECOMMENDATIONS")
    print("-" * 80)
    
    # Sample rate comparison
    if original_meta["format"]["sample_rate"] != transformed_meta["format"]["sample_rate"]:
        print(f"WARNING: Sample rate mismatch: {original_meta['format']['sample_rate']}Hz vs {transformed_meta['format']['sample_rate']}Hz")
        print(f"  Recommend resampling to common rate before metrics")
    
    # Duration comparison
    dur_diff = abs(original_meta["format"]["duration_sec"] - transformed_meta["format"]["duration_sec"])
    if dur_diff > 0.1:
        print(f"WARNING: Duration mismatch: {dur_diff:.2f}s difference")
        print(f"  May affect DTW alignment quality")
    
    # Amplitude comparison
    amp_orig = original_meta["amplitude"]["rms_mean"]
    amp_trans = transformed_meta["amplitude"]["rms_mean"]
    if abs(amp_orig - amp_trans) / amp_orig > 0.3:
        print(f"WARNING: Amplitude mismatch: {amp_orig:.3f} vs {amp_trans:.3f}")
        print(f"  Recommend level normalization before metrics")
    
    # Pitch comparison
    if original_meta["pitch"]["f0_mean_hz"] and transformed_meta["pitch"]["f0_mean_hz"]:
        f0_change = transformed_meta["pitch"]["f0_mean_hz"] / original_meta["pitch"]["f0_mean_hz"]
        print(f"\nINFO: Pitch shift: {original_meta['pitch']['f0_mean_hz']:.1f}Hz to {transformed_meta['pitch']['f0_mean_hz']:.1f}Hz")
        print(f"  Ratio: {f0_change:.2f}x ({'+' if f0_change > 1 else ''}{(f0_change-1)*100:.1f}%)")
    
    # Quality comparison
    snr_diff = transformed_meta["quality"]["snr_estimate_db"] - original_meta["quality"]["snr_estimate_db"]
    status = "INFO" if snr_diff > -3 else "WARNING"
    print(f"\n{status}: SNR change: {snr_diff:+.1f}dB")
    if snr_diff < -3:
        print(f"  Transformed audio is noisier")
    
    return original_meta, transformed_meta


def print_metadata(meta: dict):
    """Pretty print metadata."""
    print(f"File: {meta['file']}")
    print(f"\nFormat:")
    print(f"  Sample Rate: {meta['format']['sample_rate']} Hz")
    print(f"  Duration: {meta['format']['duration_sec']} sec")
    print(f"  Samples: {meta['format']['num_samples']:,}")
    
    print(f"\nAmplitude:")
    print(f"  Peak: {meta['amplitude']['max']:.4f}")
    print(f"  RMS: {meta['amplitude']['rms_mean']:.4f} +/- {meta['amplitude']['rms_std']:.4f}")
    print(f"  Dynamic Range: {meta['amplitude']['dynamic_range_db']:.1f} dB")
    if meta['amplitude']['clipped']:
        print(f"  WARNING: Clipping: {meta['amplitude']['clipping_pct']:.2f}%")
    
    print(f"\nPitch:")
    if meta['pitch']['f0_mean_hz']:
        print(f"  F0: {meta['pitch']['f0_mean_hz']:.1f} Hz (±{meta['pitch']['f0_std_hz']:.1f})")
        print(f"  Range: {meta['pitch']['f0_min_hz']:.1f} - {meta['pitch']['f0_max_hz']:.1f} Hz")
        print(f"  Voiced: {meta['pitch']['voiced_percentage']:.1f}%")
    
    print(f"\nQuality:")
    print(f"  SNR: {meta['quality']['snr_estimate_db']:.1f} dB")
    print(f"  Voice Activity: {meta['quality']['voice_activity_duration_sec']:.2f}s ({100-meta['quality']['silence_percentage']:.1f}%)")
    
    if meta['recommendations']:
        print(f"\nRecommendations:")
        for rec in meta['recommendations']:
            print(f"  {rec}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python analyze_audio_metadata.py <original.wav> <transformed.wav>")
        sys.exit(1)
    
    compare_audio_pair(sys.argv[1], sys.argv[2])
