"""
Chunk-level RVC inference with per-chunk timing measurement.

Splits source audio into fixed-length chunks (simulating real-time streaming),
processes each through Applio's RVC pipeline, and records per-chunk latency.
Outputs a converted .wav file and a .perf.json sidecar with timing data.

Usage:
    python infer_with_timing.py \
        --source assets/input/clip_001.wav \
        --output assets/output/clip_001.wav \
        --model Applio/logs/DonaldTrump/DonaldTrump.pth \
        --index Applio/logs/DonaldTrump/added_IVF256_Flat_nprobe_1_DonaldTrump_v2.index \
<<<<<<< HEAD
        --read-chunk-size 192
=======
        --read-chunk-size 192 \
        --precision fp32
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

Chunk length conditions (from proposal Table 1):
    C1: --read-chunk-size 24   ->  ~64 ms
    C2: --read-chunk-size 72   -> ~192 ms
    C3: --read-chunk-size 192  -> ~512 ms  (default / baseline)
    C4: --read-chunk-size 384  -> ~1024 ms
    C5: --read-chunk-size 768  -> ~2048 ms
<<<<<<< HEAD
=======

Precision conditions:
    fp32: float32 on CUDA (default, baseline)
    fp16: half-precision on CUDA (faster, less VRAM, ~identical quality)
    int8: dynamic quantization on CPU (torch.quantization.quantize_dynamic)
          Note: PyTorch int8 dynamic quantization requires CPU.
          Models are moved to CPU for this condition.
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

# Add Applio to sys.path so its internal imports resolve
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APPLIO_DIR = os.path.join(SCRIPT_DIR, "Applio")
if APPLIO_DIR not in sys.path:
    sys.path.insert(0, APPLIO_DIR)


APPLIO_IO_SR = 48_000   # Applio's audio I/O sample rate
INTERNAL_SR = 16_000     # Applio's internal processing sample rate


<<<<<<< HEAD
=======
def apply_precision(vc, precision: str) -> None:
    """
    Cast the loaded RVC models to the requested numeric precision.

    fp32 — no-op.

    fp16 — Cast weights to half.  The pipeline call site uses torch.autocast
           so all internal ops run in float16.  Forward hooks don't work here
           because pipeline.py calls net_g.infer() (a named method), not
           net_g() — hooks only fire on __call__.

    int8 — Move models to CPU, update config, re-instantiate Pipeline so it
           reads the new device, then quantize ONLY hubert_model.
           net_g not quantized: HiFi-GAN NSF accesses l_linear.weight.dtype
           at runtime and quantize_dynamic replaces it with a function,
           causing AttributeError.
    """
    import torch
    from rvc.infer.pipeline import Pipeline as VC

    if precision == "fp32":
        return

    if precision == "fp16":
        vc.net_g = vc.net_g.half()
        vc.hubert_model = vc.hubert_model.half()
        return

    if precision == "int8":
        vc.net_g = vc.net_g.cpu()
        vc.hubert_model = vc.hubert_model.cpu()
        vc.config.device = "cpu"
        vc.config.is_half = False
        vc.vc = VC(vc.tgt_sr, vc.config)
        vc.hubert_model = torch.quantization.quantize_dynamic(
            vc.hubert_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return

    raise ValueError(f"Unknown precision '{precision}'. Use fp32, fp16, or int8.")


>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
def infer_with_timing(
    source_path: str,
    output_path: str,
    model_path: str,
    index_path: str,
    read_chunk_size: int = 192,
    pitch: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
    embedder_model: str = "contentvec",
    target_speaker: str | None = None,
<<<<<<< HEAD
=======
    precision: str = "fp32",
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
) -> dict:
    """Run RVC inference with fixed-length chunking and per-chunk timing.

    Args:
        source_path: Path to source audio file.
        output_path: Path to save converted audio (.wav).
        model_path: Path to RVC model weights (.pth).
        index_path: Path to FAISS index file (.index).
        read_chunk_size: Applio read_chunk_size parameter.
            block_frame = read_chunk_size * 128 samples at 48kHz.
        pitch: Pitch shift in semitones.
        f0_method: F0 extraction method (rmvpe, crepe, fcpe).
        index_rate: Speaker embedding retrieval blend rate.
        embedder_model: Feature extractor model name.
        target_speaker: Label for the target speaker (metadata only).
<<<<<<< HEAD
=======
        precision: fp32 (default), fp16, or int8.
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

    Returns:
        Dict with performance statistics.
    """
    import torch
    from rvc.infer.infer import VoiceConverter
    from rvc.lib.utils import load_audio_infer
<<<<<<< HEAD
    
=======

>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
    # Resolve absolute paths BEFORE any working directory changes
    source_path = os.path.abspath(source_path)
    output_path = os.path.abspath(output_path)
    model_path = os.path.abspath(model_path)
    index_path = os.path.abspath(index_path)

    # Compute chunk sizes
    block_frame_48k = read_chunk_size * 128              # samples at 48kHz
    block_frame_ms = block_frame_48k / APPLIO_IO_SR * 1000
    block_frame_16k = int(block_frame_48k * INTERNAL_SR / APPLIO_IO_SR)

    print(f"  Chunk config  : read_chunk_size={read_chunk_size}, "
          f"block_frame={block_frame_48k} samples, ~{block_frame_ms:.0f} ms")
<<<<<<< HEAD
=======
    print(f"  Precision     : {precision}")
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

    # Save current working directory so we can restore it
    original_cwd = os.getcwd()

<<<<<<< HEAD
=======
    # Determine if we are running on CUDA after precision is applied.
    # int8 forces CPU; fp32/fp16 use whatever device Applio's config selects.
    using_cuda = (precision != "int8") and torch.cuda.is_available()

>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
    try:
        # Applio expects to be run from its own directory (for config JSON loading)
        os.chdir(APPLIO_DIR)

<<<<<<< HEAD
        # Initialize voice converter and load model
=======
        # Initialize voice converter and load model at default fp32 on device
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
        vc = VoiceConverter()
        vc.get_vc(model_path, 0)

        if not vc.hubert_model or embedder_model != vc.last_embedder_model:
            vc.load_hubert(embedder_model)
            vc.last_embedder_model = embedder_model

<<<<<<< HEAD
=======
        # Apply precision AFTER Applio has finished loading (which always uses fp32)
        apply_precision(vc, precision)

>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
        # Load audio at 16kHz (Applio's internal processing rate)
        audio = load_audio_infer(source_path, INTERNAL_SR)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        # Prepare index path
        file_index = (
            index_path.strip().strip('"').strip("\n")
            .strip('"').strip().replace("trained", "added")
        )

        # Split into fixed-length chunks at 16kHz
        num_samples = len(audio)
        chunks = []
        for start in range(0, num_samples, block_frame_16k):
            chunk = audio[start:start + block_frame_16k]
            # Zero-pad the last chunk if shorter than block_frame
            if len(chunk) < block_frame_16k:
                chunk = np.pad(chunk, (0, block_frame_16k - len(chunk)))
            chunks.append(chunk)

        print(f"  Source audio  : {num_samples} samples @ {INTERNAL_SR} Hz "
              f"({num_samples / INTERNAL_SR:.2f}s)")
        print(f"  Chunks        : {len(chunks)} x {block_frame_16k} samples")

<<<<<<< HEAD
        # Reset GPU memory stats
        if torch.cuda.is_available():
=======
        # Reset GPU memory stats only when actually using CUDA
        if using_cuda:
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
            torch.cuda.reset_peak_memory_stats()

        # Process each chunk with timing
        chunk_latencies_ms: list[float] = []
        converted_chunks: list[np.ndarray] = []

        for i, chunk in enumerate(chunks):
<<<<<<< HEAD
            if torch.cuda.is_available():
=======
            # Synchronize before timing only when on CUDA —
            # otherwise the timer just measures CPU dispatch, not compute.
            if using_cuda:
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
                torch.cuda.synchronize()

            t0 = time.perf_counter()

<<<<<<< HEAD
            audio_opt = vc.vc.pipeline(
                model=vc.hubert_model,
                net_g=vc.net_g,
                sid=0,
                audio=chunk,
                pitch=pitch,
                f0_method=f0_method,
                file_index=file_index,
                index_rate=index_rate,
                pitch_guidance=vc.use_f0,
                volume_envelope=1.0,
                version=vc.version,
                protect=0.5,
                f0_autotune=False,
                f0_autotune_strength=1.0,
                proposed_pitch=False,
                proposed_pitch_threshold=155.0,
            )

            if torch.cuda.is_available():
=======
            # autocast handles dtype promotion for all ops in the call tree.
            # For fp16 on CUDA: pipeline.py hard-codes feats.float() before
            # calling net_g.infer() — autocast overrides that and keeps
            # float16 throughout without touching Applio source.
            # For fp32/int8: autocast is a no-op (int8 runs on CPU).
            autocast_device = "cuda" if using_cuda else "cpu"
            autocast_enabled = (precision == "fp16")
            with torch.autocast(device_type=autocast_device, enabled=autocast_enabled):
                audio_opt = vc.vc.pipeline(
                    model=vc.hubert_model,
                    net_g=vc.net_g,
                    sid=0,
                    audio=chunk,
                    pitch=pitch,
                    f0_method=f0_method,
                    file_index=file_index,
                    index_rate=index_rate,
                    pitch_guidance=vc.use_f0,
                    volume_envelope=1.0,
                    version=vc.version,
                    protect=0.5,
                    f0_autotune=False,
                    f0_autotune_strength=1.0,
                    proposed_pitch=False,
                    proposed_pitch_threshold=155.0,
                )

            if using_cuda:
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000
            chunk_latencies_ms.append(latency_ms)
            converted_chunks.append(audio_opt)

            if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                print(f"  Chunk {i + 1}/{len(chunks)} — {latency_ms:.1f} ms")

        # Concatenate output
        output_audio = np.concatenate(converted_chunks)

        # Clip to prevent clipping
        audio_max = np.abs(output_audio).max() / 0.99
        if audio_max > 1:
            output_audio /= audio_max

<<<<<<< HEAD
        # Peak GPU memory
        peak_gpu_mb = None
        if torch.cuda.is_available():
=======
        # Peak GPU memory — only meaningful for CUDA runs
        peak_gpu_mb = None
        if using_cuda:
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
            peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    finally:
        os.chdir(original_cwd)

    # Save output audio
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, output_audio, vc.tgt_sr, format="WAV")

    # Build performance stats
    latencies = np.array(chunk_latencies_ms)
    perf_stats = {
        "num_chunks": len(chunks),
<<<<<<< HEAD
=======
        "precision": precision,
        "device": "cpu" if precision == "int8" else str(vc.config.device),
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
        "chunk_latencies_ms": [round(v, 3) for v in chunk_latencies_ms],
        "mean_chunk_latency_ms": round(float(latencies.mean()), 3),
        "min_chunk_latency_ms": round(float(latencies.min()), 3),
        "max_chunk_latency_ms": round(float(latencies.max()), 3),
        "std_chunk_latency_ms": round(float(latencies.std()), 3),
        "total_latency_ms": round(float(latencies.sum()), 3),
        "peak_gpu_memory_mb": round(peak_gpu_mb, 2) if peak_gpu_mb else None,
        "block_frame_samples": block_frame_48k,
        "block_frame_ms": round(block_frame_ms, 1),
        "source_file": os.path.basename(source_path),
        "target_speaker": target_speaker,
        "read_chunk_size": read_chunk_size,
    }

    # Save perf stats as sidecar JSON
    perf_path = os.path.splitext(output_path)[0] + ".perf.json"
    with open(perf_path, "w") as f:
        json.dump(perf_stats, f, indent=2)

    print(f"  Output        : {output_path}")
    print(f"  Perf stats    : {perf_path}")
    print(f"  Mean latency  : {perf_stats['mean_chunk_latency_ms']:.1f} ms/chunk")
    if peak_gpu_mb:
        print(f"  Peak GPU mem  : {peak_gpu_mb:.1f} MB")

    return perf_stats


def main():
    parser = argparse.ArgumentParser(
        description="RVC inference with per-chunk timing measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Chunk length conditions (proposal Table 1):
  C1:  --read-chunk-size  24  ->   ~64 ms  (very short)
  C2:  --read-chunk-size  72  ->  ~192 ms  (short, near real-time threshold)
  C3:  --read-chunk-size 192  ->  ~512 ms  (default / baseline)
  C4:  --read-chunk-size 384  -> ~1024 ms  (long)
  C5:  --read-chunk-size 768  -> ~2048 ms  (very long)
<<<<<<< HEAD
=======

Precision conditions:
  fp32  float32 on CUDA (default, baseline)
  fp16  half-precision on CUDA (faster, less VRAM)
  int8  dynamic quantization on CPU (torch.quantization.quantize_dynamic)
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
        """,
    )

    parser.add_argument("--source", required=True, help="Path to source audio file")
    parser.add_argument("--output", required=True, help="Path to save converted audio (.wav)")
    parser.add_argument("--model", required=True, help="Path to RVC model weights (.pth)")
    parser.add_argument("--index", required=True, help="Path to FAISS index file (.index)")
    parser.add_argument(
        "--read-chunk-size", type=int, default=192, dest="read_chunk_size",
        help="Applio read_chunk_size parameter (default: 192 = C3 baseline)",
    )
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones")
    parser.add_argument(
        "--f0-method", type=str, default="rmvpe", dest="f0_method",
        choices=["rmvpe", "crepe", "crepe-tiny", "fcpe"],
        help="F0 extraction method (default: rmvpe)",
    )
    parser.add_argument(
        "--index-rate", type=float, default=0.75, dest="index_rate",
        help="Index rate for speaker embedding retrieval (default: 0.75)",
    )
    parser.add_argument(
        "--embedder-model", type=str, default="contentvec", dest="embedder_model",
        help="Feature extractor model (default: contentvec)",
    )
    parser.add_argument(
        "--target-speaker", type=str, default=None, dest="target_speaker",
        help="Target speaker label (metadata only, written to perf.json)",
    )
<<<<<<< HEAD
=======
    parser.add_argument(
        "--precision", type=str, default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="Model precision: fp32 (default), fp16 (CUDA half), int8 (CPU dynamic quant)",
    )
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

    args = parser.parse_args()

    # Validate input files
    if not os.path.isfile(args.source):
        print(f"Error: source file not found: {args.source}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.index):
        print(f"Error: index file not found: {args.index}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  RVC Inference with Chunk-Level Timing")
    print("=" * 60)

    infer_with_timing(
        source_path=args.source,
        output_path=args.output,
        model_path=args.model,
        index_path=args.index,
        read_chunk_size=args.read_chunk_size,
        pitch=args.pitch,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        embedder_model=args.embedder_model,
        target_speaker=args.target_speaker,
<<<<<<< HEAD
=======
        precision=args.precision,
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
    )


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
