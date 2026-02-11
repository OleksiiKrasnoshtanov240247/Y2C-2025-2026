"""
Speech Intelligibility metrics for baseline audio evaluation.

Metrics:
- WER: Word Error Rate (via Whisper large-v3 + jiwer)
- CER: Character Error Rate (via Whisper large-v3 + jiwer)
- STOI: Short-Time Objective Intelligibility (via pystoi)

Reference: audio_evaluation_metrics.md v1.0
"""

import numpy as np
from tqdm import tqdm


def transcribe_audio_whisper(
    audio: np.ndarray,
    sr: int,
    model_name: str = "large-v3-turbo",
    device: str = "cpu",
    compute_type: str = "int8",
) -> str:
    """Transcribe audio using faster-whisper.

    Args:
        audio: Audio signal.
        sr: Sample rate.
        model_name: Whisper model name.
        device: Device for inference.
        compute_type: Quantization type.

    Returns:
        Transcribed text.
    """
    from faster_whisper import WhisperModel
    
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    
    segments, info = model.transcribe(
        audio.astype(np.float32),
        language="en",
        beam_size=5,
        vad_filter=True,  # Voice activity detection for better accuracy
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    
    # Calculate estimated segments
    duration = info.duration
    estimated_segments = max(1, int(duration / 5))
    
    text_parts = []
    with tqdm(total=estimated_segments, desc="  Transcribing", unit="seg", leave=False) as pbar:
        for segment in segments:
            text_parts.append(segment.text)
            pbar.update(1)
        pbar.total = len(text_parts)
        pbar.refresh()
    
    return " ".join(text_parts).strip()


def compute_wer_cer_single(
    audio: np.ndarray,
    sr: int,
    ground_truth_transcript: str,
    whisper_model_name: str = "large-v3-turbo",
    device: str = "cpu",
    compute_type: str = "int8",
) -> dict:
    """Compute WER/CER for a single audio file against ground truth.

    This is for voice conversion: measure how well transformed audio
    preserves the original content.

    Args:
        audio: Audio signal to transcribe.
        sr: Sample rate.
        ground_truth_transcript: Reference transcript.
        whisper_model_name: Whisper model.
        device: Device for inference.
        compute_type: Quantization type.

    Returns:
        Dict with 'wer', 'cer', 'reference_transcript', 'hypothesis_transcript'.
    """
    import jiwer
    
    # Transcribe the audio
    hypothesis = transcribe_audio_whisper(
        audio, sr,
        model_name=whisper_model_name,
        device=device,
        compute_type=compute_type
    )
    
    reference = ground_truth_transcript.strip()
    
    if not reference:
        return {
            "wer": float("nan"),
            "cer": float("nan"),
            "reference_transcript": reference,
            "hypothesis_transcript": hypothesis,
        }
    
    # Apply standard text normalization for cleaner comparison
    def normalize_text(text: str) -> str:
        """Normalize text for WER/CER calculation."""
        text = text.lower()
        text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
        text = " ".join(text.split())  # Remove multiple spaces
        return text
    
    ref_normalized = normalize_text(reference)
    hyp_normalized = normalize_text(hypothesis)
    
    werval = jiwer.wer(ref_normalized, hyp_normalized)
    cerval = jiwer.cer(ref_normalized, hyp_normalized)
    
    return {
        "wer": float(werval),
        "cer": float(cerval),
        "reference_transcript": reference,
        "hypothesis_transcript": hypothesis,
    }


def compute_wer_cer(
    original: np.ndarray,
    transformed: np.ndarray,
    sr: int,
    whisper_model_name: str = "large-v3",
    ground_truth_transcript: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
) -> dict:
    """Compute Word Error Rate (WER) and Character Error Rate (CER).

    For baseline evaluation: transcribes original audio with Whisper to obtain
    ground truth, then transcribes transformed audio and compares.
    Optionally accepts a manual ground truth transcript.

    Uses faster-whisper for improved performance via CTranslate2.

    Args:
        original: Original audio signal.
        transformed: Transformed audio signal.
        sr: Sample rate (should be 16000).
        whisper_model_name: Whisper model size. Default 'large-v3' per
            standardisation requirements.
        ground_truth_transcript: Optional manual ground truth. If None,
            the original audio is transcribed automatically.
        device: Device to run inference on ("cpu", "cuda", "auto").
        compute_type: Quantization type ("int8", "float16", "float32").
            "int8" is faster and default for CPU.

    Returns:
        Dict with 'wer', 'cer', 'reference_transcript', 'hypothesis_transcript'.
    """
    from faster_whisper import WhisperModel
    import jiwer

    # Load model (faster-whisper uses CTranslate2 for performance)
    model = WhisperModel(
        whisper_model_name,
        device=device,
        compute_type=compute_type,
    )

    def transcribe_audio(audio: np.ndarray, desc: str = "Transcribing") -> str:
        """Transcribe audio using faster-whisper and return full text."""
        segments, info = model.transcribe(
            audio.astype(np.float32),
            language="en",
            beam_size=5,
        )
        
        # Calculate estimated number of segments based on audio duration
        duration = info.duration
        estimated_segments = max(1, int(duration / 5))  # Rough estimate: 5 seconds per segment
        
        # Concatenate all segment texts with progress bar
        text_parts = []
        with tqdm(total=estimated_segments, desc=desc, unit="seg", leave=False) as pbar:
            for segment in segments:
                text_parts.append(segment.text)
                pbar.update(1)
            # Update to actual total in case estimate was off
            pbar.total = len(text_parts)
            pbar.refresh()
        
        text = " ".join(text_parts).strip()
        return text

    # Get ground truth transcript
    if ground_truth_transcript is None:
        reference = transcribe_audio(original, desc="  → Transcribing original audio")
    else:
        reference = ground_truth_transcript.strip()

    # Transcribe transformed audio
    hypothesis = transcribe_audio(transformed, desc="  → Transcribing transformed audio")

    if not reference:
        return {
            "wer": float("nan"),
            "cer": float("nan"),
            "reference_transcript": reference,
            "hypothesis_transcript": hypothesis,
        }

    wer = jiwer.wer(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)

    return {
        "wer": float(wer),
        "cer": float(cer),
        "reference_transcript": reference,
        "hypothesis_transcript": hypothesis,
    }


def compute_stoi(
    original: np.ndarray,
    transformed: np.ndarray,
    sr: int,
) -> float:
    """Compute Short-Time Objective Intelligibility (STOI).

    Requires paired reference signal (original clean audio).
    Both signals must be the same length and sample rate.

    Args:
        original: Original clean audio signal.
        transformed: Transformed audio signal.
        sr: Sample rate (should be 16000).

    Returns:
        STOI score in [0, 1]. Higher is better.
    """
    from pystoi import stoi

    # pystoi requires equal-length signals
    min_len = min(len(original), len(transformed))
    original = original[:min_len]
    transformed = transformed[:min_len]

    score = stoi(original, transformed, sr, extended=False)
    return float(score)
