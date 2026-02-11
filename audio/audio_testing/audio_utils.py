"""
Audio utility functions for loading, resampling, and aligning audio signals.

Standardisation (per metrics reference document):
- All audio resampled to 16 kHz before metric computation
- WAV PCM 16-bit for metric computation
- Smart preprocessing based on original audio characteristics
"""

import numpy as np
import librosa
import soundfile as sf

STANDARD_SR = 16_000  # 16 kHz standard for speech processing


def load_audio(path: str, sr: int = STANDARD_SR) -> tuple[np.ndarray, int]:
    """Load audio file and resample to the standard sample rate.

    Supports WAV and MP3 formats. Converts to mono if stereo.

    Returns:
        Tuple of (audio_signal, sample_rate).
    """
    audio, orig_sr = librosa.load(path, sr=sr, mono=True)
    return audio, sr


def prepare_audio_for_metrics(path: str) -> tuple[np.ndarray, int, dict]:
    """Load and preprocess audio with intelligent handling.
    
    Unlike basic load_audio(), this function:
    - Analyzes original audio metadata
    - Applies level normalization if needed
    - Chooses optimal resampling strategy based on source quality
    - Preserves as much quality as possible
    
    Returns:
        Tuple of (processed_audio, sample_rate, metadata_dict)
    """
    # Load raw audio to analyze
    audio_raw, sr_original = sf.read(path)
    
    # Convert stereo to mono if needed
    if len(audio_raw.shape) > 1:
        audio_raw = np.mean(audio_raw, axis=1)
    
    metadata = {
        "original_sr": sr_original,
        "original_duration": len(audio_raw) / sr_original,
        "original_peak": float(np.max(np.abs(audio_raw))),
        "original_rms": float(np.sqrt(np.mean(audio_raw ** 2))),
    }
    
    # Level normalization (prevent clipping, standardize loudness)
    # Target RMS around -20 dB (0.1) for consistent comparison
    target_rms = 0.1
    current_rms = metadata["original_rms"]
    
    if current_rms > 1e-6:  # Avoid division by zero
        normalization_gain = target_rms / current_rms
        # Cap gain to prevent excessive amplification of noise
        normalization_gain = min(normalization_gain, 3.0)
        audio_normalized = audio_raw * normalization_gain
        metadata["normalization_applied"] = True
        metadata["normalization_gain_db"] = 20 * np.log10(normalization_gain)
    else:
        audio_normalized = audio_raw
        metadata["normalization_applied"] = False
    
    # Prevent clipping after normalization
    peak = np.max(np.abs(audio_normalized))
    if peak > 0.95:
        audio_normalized = audio_normalized * (0.95 / peak)
        metadata["peak_limiting_applied"] = True
    
    # Smart resampling
    if sr_original != STANDARD_SR:
        # Use scipy for reliable resampling (included with numpy/scipy)
        audio_resampled = librosa.resample(
            audio_normalized,
            orig_sr=sr_original,
            target_sr=STANDARD_SR,
            res_type='scipy'  # Reliable, fast SciPy resampling
        )
        metadata["resampled"] = True
    else:
        audio_resampled = audio_normalized
        metadata["resampled"] = False
    
    return audio_resampled, STANDARD_SR, metadata


def trim_or_pad_to_match(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim or zero-pad the target signal to match the reference length.

    Required for metrics like STOI that need equal-length signals.
    """
    ref_len = len(reference)
    tgt_len = len(target)

    if tgt_len > ref_len:
        target = target[:ref_len]
    elif tgt_len < ref_len:
        target = np.pad(target, (0, ref_len - tgt_len), mode="constant")

    return reference, target
