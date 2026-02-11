"""
Speaker Similarity metrics for baseline audio evaluation.

Metrics:
- SECS: Speaker Embedding Cosine Similarity (via ECAPA-TDNN)
- MCD: Mel Cepstral Distortion (via librosa MFCC + DTW)
- F0 Correlation & F0 RMSE (via librosa pyin)

Reference: audio_evaluation_metrics.md v1.0
"""

import numpy as np
import librosa
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

# Compatibility fix for torchaudio 2.1+ with speechbrain
# torchaudio removed list_audio_backends() but speechbrain 1.0.x still tries to call it
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: [""]  # type: ignore

# Compatibility fix for huggingface_hub 0.20+ with speechbrain
# huggingface_hub renamed use_auth_token to token parameter
import huggingface_hub
from functools import wraps

_original_hf_hub_download = huggingface_hub.hf_hub_download

@wraps(_original_hf_hub_download)
def _patched_hf_hub_download(*args, **kwargs):
    # Convert use_auth_token to token for backwards compatibility
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    
    # Handle missing custom.py file gracefully
    try:
        return _original_hf_hub_download(*args, **kwargs)
    except huggingface_hub.errors.EntryNotFoundError as e:
        # If custom.py is missing, create a placeholder and return its path
        filename = kwargs.get('filename', args[1] if len(args) > 1 else None)
        if filename and "custom.py" in filename:
            import os
            import tempfile
            # Create temp placeholder
            temp_file = os.path.join(tempfile.gettempdir(), "custom.py")
            if not os.path.exists(temp_file):
                with open(temp_file, "w") as f:
                    f.write("# Placeholder for missing custom.py\n")
            return temp_file
        raise

huggingface_hub.hf_hub_download = _patched_hf_hub_download

# MFCC configuration from standardisation requirements:
# 24 coefficients, 25ms window, 10ms hop, 80 mel channels
MCD_N_MFCC = 24
MCD_WIN_LENGTH_SEC = 0.025
MCD_HOP_LENGTH_SEC = 0.010
MCD_N_MELS = 80


def compute_secs(
    original: np.ndarray,
    transformed: np.ndarray,
    sr: int,
) -> float:
    """Compute Speaker Embedding Cosine Similarity (SECS).

    Uses SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) as the
    standardised speaker encoder.

    Args:
        original: Original audio signal (1-D numpy array).
        transformed: Transformed audio signal (1-D numpy array).
        sr: Sample rate (should be 16000).

    Returns:
        Cosine similarity score in [-1, 1]. Higher is better.
    """
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
        savedir="tmp_models/spkrec-ecapa-voxceleb",
    )

    orig_tensor = torch.tensor(original, dtype=torch.float32).unsqueeze(0)
    trans_tensor = torch.tensor(transformed, dtype=torch.float32).unsqueeze(0)

    emb_original = classifier.encode_batch(orig_tensor).squeeze().detach().numpy()
    emb_transformed = classifier.encode_batch(trans_tensor).squeeze().detach().numpy()

    similarity = 1.0 - cosine(emb_original, emb_transformed)
    return float(similarity)


def compute_mcd(
    original: np.ndarray,
    transformed: np.ndarray,
    sr: int,
) -> float:
    """Compute Mel Cepstral Distortion (MCD) with DTW alignment.

    Uses standardised MFCC configuration: 24 coefficients, 25ms window,
    10ms hop, 80 mel filterbank channels.

    Args:
        original: Original audio signal (1-D numpy array).
        transformed: Transformed audio signal (1-D numpy array).
        sr: Sample rate (should be 16000).

    Returns:
        MCD value in dB. Lower is better.
    """
    from dtw import dtw as dtw_align
    import scipy

    win_length = int(MCD_WIN_LENGTH_SEC * sr)
    hop_length = int(MCD_HOP_LENGTH_SEC * sr)

    # Compute mel spectrograms
    mel_orig = librosa.feature.melspectrogram(
        y=original, sr=sr,
        n_mels=MCD_N_MELS,
        win_length=win_length,
        hop_length=hop_length,
    )

    mel_trans = librosa.feature.melspectrogram(
        y=transformed, sr=sr,
        n_mels=MCD_N_MELS,
        win_length=win_length,
        hop_length=hop_length,
    )

    # Apply natural log (not dB scale) for standard MCD computation
    log_mel_orig = np.log(mel_orig + 1e-10)
    log_mel_trans = np.log(mel_trans + 1e-10)

    # Compute MFCCs via DCT
    mfcc_orig = scipy.fft.dct(log_mel_orig, type=2, axis=0, norm='ortho')[:MCD_N_MFCC].T
    mfcc_trans = scipy.fft.dct(log_mel_trans, type=2, axis=0, norm='ortho')[:MCD_N_MFCC].T

    # DTW alignment — skip c0 (energy) coefficient
    mfcc_orig = mfcc_orig[:, 1:]
    mfcc_trans = mfcc_trans[:, 1:]

    alignment = dtw_align(mfcc_orig, mfcc_trans)
    aligned_orig = mfcc_orig[alignment.index1]
    aligned_trans = mfcc_trans[alignment.index2]

    # MCD formula: (10 / ln(10)) * sqrt(2 * mean((c_k - c'_k)^2))
    # Note: Using mean instead of sum to normalize by number of coefficients
    diff = aligned_orig - aligned_trans
    frame_mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(diff ** 2, axis=1))
    mcd = float(np.mean(frame_mcd))

    return mcd


def compute_f0_metrics(
    original: np.ndarray,
    transformed: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 600.0,
) -> dict:
    """Compute F0 (pitch) metrics: Pearson Correlation and RMSE.

    Uses librosa pyin for pitch extraction.

    Args:
        original: Original audio signal.
        transformed: Transformed audio signal.
        sr: Sample rate.
        fmin: Minimum expected F0 in Hz.
        fmax: Maximum expected F0 in Hz.

    Returns:
        Dict with 'f0_correlation' and 'f0_rmse' keys.
    """
    f0_orig, voiced_orig, _ = librosa.pyin(
        original, fmin=fmin, fmax=fmax, sr=sr
    )
    f0_trans, voiced_trans, _ = librosa.pyin(
        transformed, fmin=fmin, fmax=fmax, sr=sr
    )

    # Align lengths (pyin can produce slightly different frame counts)
    min_len = min(len(f0_orig), len(f0_trans))
    f0_orig = f0_orig[:min_len]
    f0_trans = f0_trans[:min_len]
    voiced_orig = voiced_orig[:min_len]
    voiced_trans = voiced_trans[:min_len]

    # Only compare frames where both signals are voiced
    both_voiced = voiced_orig & voiced_trans

    if np.sum(both_voiced) < 2:
        return {
            "f0_correlation": float("nan"),
            "f0_rmse": float("nan"),
            "voiced_frames": int(np.sum(both_voiced)),
        }

    f0_o = f0_orig[both_voiced]
    f0_t = f0_trans[both_voiced]

    correlation, _ = pearsonr(f0_o, f0_t)
    rmse = float(np.sqrt(np.mean((f0_o - f0_t) ** 2)))

    return {
        "f0_correlation": float(correlation),
        "f0_rmse": rmse,
        "voiced_frames": int(np.sum(both_voiced)),
    }
