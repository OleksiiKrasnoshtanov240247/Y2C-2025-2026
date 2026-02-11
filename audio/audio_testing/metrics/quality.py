"""
Perceptual Quality metrics for voice conversion evaluation.

Metrics:
- PESQ: Perceptual Evaluation of Speech Quality
- SNR: Signal-to-Noise Ratio estimate
- Loudness normalization utilities
"""

import numpy as np
import librosa


def compute_pesq(
    reference: np.ndarray,
    degraded: np.ndarray,
    sr: int,
    mode: str = "wb",
) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality).

    PESQ is an ITU standard for measuring speech quality.
    
    Args:
        reference: Reference (clean) audio signal.
        degraded: Degraded/processed audio signal.
        sr: Sample rate (8000 for narrowband, 16000 for wideband).
        mode: 'nb' for narrowband (8kHz) or 'wb' for wideband (16kHz).

    Returns:
        PESQ score. Range depends on mode:
        - Narrowband: 1.0 to 4.5
        - Wideband: 1.0 to 4.5
        Higher is better.
    """
    from pesq import pesq
    
    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    # PESQ requires specific sample rates
    if sr == 8000:
        mode = "nb"
    elif sr == 16000:
        mode = "wb"
    else:
        # Resample to 16kHz for wideband PESQ
        reference = librosa.resample(reference, orig_sr=sr, target_sr=16000)
        degraded = librosa.resample(degraded, orig_sr=sr, target_sr=16000)
        sr = 16000
        mode = "wb"
    
    score = pesq(sr, reference, degraded, mode)
    return float(score)


def compute_snr(
    audio: np.ndarray,
    top_db_percentile: int = 60,
) -> float:
    """Estimate Signal-to-Noise Ratio of audio.

    Uses a simple heuristic: signal is the top 60% percentile of RMS energy,
    noise is the bottom 10% percentile.

    Args:
        audio: Audio signal.
        top_db_percentile: Percentile for signal estimation (default 60).

    Returns:
        Estimated SNR in dB.
    """
    # Compute frame-wise RMS
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Signal estimate: upper percentile
    signal_rms = np.percentile(rms, top_db_percentile)
    
    # Noise estimate: lower percentile
    noise_rms = np.percentile(rms, 10)
    
    # Avoid division by zero
    if noise_rms < 1e-10:
        return float('inf')
    
    snr = 20 * np.log10(signal_rms / noise_rms)
    return float(snr)


def normalize_loudness(
    audio: np.ndarray,
    target_loudness: float = -23.0,
) -> np.ndarray:
    """Normalize audio to target loudness (ITU-R BS.1770 LUFS).

    Uses pyloudnorm for perceptually-motivated loudness normalization.

    Args:
        audio: Audio signal.
        target_loudness: Target loudness in LUFS (default -23.0, broadcast standard).

    Returns:
        Normalized audio.
    """
    try:
        import pyloudnorm as pyln
        
        # Measure loudness
        meter = pyln.Meter(16000)  # Assumes 16kHz
        loudness = meter.integrated_loudness(audio)
        
        # Normalize
        normalized = pyln.normalize.loudness(audio, loudness, target_loudness)
        return normalized
    except ImportError:
        # Fallback: simple RMS normalization
        target_rms = 10 ** (target_loudness / 20)
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 1e-6:
            gain = target_rms / current_rms
            return audio * gain
        return audio
