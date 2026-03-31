"""
metrics.py — Group O1, BUas ADS&AI Y2C 2026
5 video quality metrics on synchronized input–output video pairs.

Usage:
    python metrics.py --generated output.mp4 --reference avatar.jpg
    python metrics.py --generated output.mp4 --reference avatar.jpg --source original.mp4
    python metrics.py --generated output.mp4 --reference avatar.jpg --perf_json clip.perf.json
"""

import contextlib
import io
import os
import cv2, json, time, argparse, sys, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")


# ── Output suppression ────────────────────────────────────────────────────────
# InsightFace, onnxruntime, and MediaPipe all print verbose C++ runtime messages
# (Applied providers, find model, GL context init) directly to file descriptors 1
# and 2, bypassing Python's sys.stdout/stderr.  The only way to suppress them is
# to redirect the underlying OS file descriptors temporarily.

@contextlib.contextmanager
def suppress_c_output():
    """
    Suppress C-level stdout and stderr (fd 1 and fd 2) for the duration of the
    block.  This catches output from C++ libraries that bypass sys.stdout.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


# ── CUDA detection ────────────────────────────────────────────────────────────
# Prepends pip-installed CUDA DLL directories to PATH so onnxruntime-gpu can
# find them before it initialises.  Then attempts to import torch; sets
# HAS_TORCH and DEVICE accordingly.

HAS_TORCH = False
DEVICE = "cpu"

try:
    import importlib.util

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None and torch_spec.origin is not None:
        site_packages = Path(torch_spec.origin).parent.parent
        new_path = os.environ.get("PATH", "")
        for nv_dir in ["cublas", "cudnn", "cuda_runtime", "cufft", "cusparse",
                       "cusolver", "curand", "nvtx"]:
            nv_bin = site_packages / "nvidia" / nv_dir / "bin"
            if nv_bin.exists():
                new_path = str(nv_bin) + os.pathsep + new_path
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(nv_bin))

        torch_lib = site_packages / "torch" / "lib"
        if torch_lib.exists():
            new_path = str(torch_lib) + os.pathsep + new_path
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(torch_lib))

        os.environ["PATH"] = new_path

        # Actually import torch — find_spec only checks existence, doesn't import
        import torch
        HAS_TORCH = True
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

except Exception as _e:
    HAS_TORCH = False
    DEVICE = "cpu"


def _get_onnx_providers():
    try:
        import onnxruntime as _ort
        available = _ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            print("  [onnxruntime] Using GPU (CUDA)")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print("  [onnxruntime] CUDA not available, using CPU")
            return ["CPUExecutionProvider"]
    except Exception:
        return ["CPUExecutionProvider"]

ONNX_PROVIDERS = _get_onnx_providers()
INSIGHTFACE_CTX = 0 if "CUDAExecutionProvider" in ONNX_PROVIDERS else -1

# ── optional deps ─────────────────────────────────────────────────────────────
# Each flag is set independently so a missing package only disables its metric,
# not everything.  Bare `pass` on a multi-import block silently breaks all
# downstream consumers — track each import separately.

HAS_INSIGHTFACE = False
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    pass

HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    pass

HAS_LPIPS = False
try:
    import lpips as _lpips
    HAS_LPIPS = True
except ImportError:
    pass

HAS_TORCHVISION = False
try:
    from torchvision import transforms as _T, models as _models
    HAS_TORCHVISION = True
except ImportError:
    pass

HAS_SCIPY = False
try:
    from scipy.linalg import sqrtm as _sqrtm
    HAS_SCIPY = True
except ImportError:
    pass


# ── helpers ───────────────────────────────────────────────────────────────────

def read_frames(path, max_frames=300):
    cap = cv2.VideoCapture(str(path))
    frames, timestamps = [], []
    while len(frames) < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    return frames, timestamps


def get_info(path):
    cap = cv2.VideoCapture(str(path))
    info = {
        "fps":    round(cap.get(cv2.CAP_PROP_FPS), 2),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_s"] = round(info["frames"] / max(info["fps"], 1), 2)
    cap.release()
    return info


def r(frame, size=(256, 256)):
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


# ── METRIC 1: Identity Preservation — ArcFace cosine similarity ──────────────

def metric_arcface(ref_image, gen_frames):
    """
    ArcFace R100 cosine similarity between reference image and each output frame.
    Uses InsightFace (RetinaFace detection + ArcFace embedding).

    Note: detection requires a minimum face resolution — frames below ~112px
    wide (e.g. MuseTalk 224x224 output with small face region) may yield zero
    detections.  The result will report no_face_frames accordingly.
    """
    if not HAS_INSIGHTFACE:
        return {"error": "pip install insightface onnxruntime-gpu"}

    with suppress_c_output():
        app = FaceAnalysis(name="buffalo_l", providers=ONNX_PROVIDERS)
        app.prepare(ctx_id=INSIGHTFACE_CTX, det_size=(640, 640))

    ref_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_faces = app.get(ref_rgb)
    if not ref_faces:
        return {"error": "No face detected in reference image"}
    ref_emb = ref_faces[0].normed_embedding  # 512-D, already L2-normalised

    sims = []
    no_face_count = 0
    for frame in gen_frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(frame_rgb)
        if not faces:
            no_face_count += 1
            continue
        emb = faces[0].normed_embedding
        sims.append(float(np.dot(ref_emb, emb)))

    if not sims:
        return {
            "error": "No faces detected in any output frame",
            "no_face_frames": no_face_count,
            "note": "Low-resolution outputs (e.g. 224x224) may not meet ArcFace detection threshold.",
        }

    a = np.array(sims)
    return {
        "mean":            round(float(np.mean(a)), 4),
        "std":             round(float(np.std(a)),  4),
        "p5":              round(float(np.percentile(a, 5)), 4),
        "no_face_frames":  no_face_count,
        "n_frames":        len(sims),
        "note": "Cosine similarity, range [-1, 1]. Higher = better identity preservation.",
    }


# ── METRIC 2: Expression & Pose Transfer — Landmark NME ──────────────────────

def metric_landmark_nme(src_frames, gen_frames):
    """
    Facial landmark NME between synchronized input and output frames.
    Uses MediaPipe FaceMesh.  NME normalised by inter-ocular distance.
    """
    if not HAS_MEDIAPIPE:
        return {"error": "pip install mediapipe"}
    if not src_frames:
        return {"error": "provide --source for NME"}

    LEFT_EYE  = 33
    RIGHT_EYE = 263
    NOSE_TIP  = 1

    def get_lm(frame, face_mesh):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm])

    nmes = []
    no_face = 0

    try:
        FaceMesh = mp.solutions.face_mesh.FaceMesh
    except AttributeError:
        return {"error": "mediapipe.solutions not available — run: uv pip install 'mediapipe==0.10.14'"}

    with suppress_c_output():
        _fm_ctx = FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    with _fm_ctx as fm:
        for sf, gf in zip(src_frames, gen_frames):
            src_lm = get_lm(sf, fm)
            gen_lm = get_lm(gf, fm)

            if src_lm is None or gen_lm is None:
                no_face += 1
                continue

            iod = np.linalg.norm(src_lm[LEFT_EYE] - src_lm[RIGHT_EYE])
            if iod < 1e-6:
                continue

            anchors = [LEFT_EYE, RIGHT_EYE, NOSE_TIP]
            sa = src_lm[anchors];  ga = gen_lm[anchors]
            sc = sa.mean(0);       gc = ga.mean(0)
            sa -= sc;              ga -= gc
            scale = np.linalg.norm(sa) / (np.linalg.norm(ga) + 1e-8)
            U, _, Vt = np.linalg.svd((ga * scale).T @ sa)
            R = Vt.T @ U.T
            gen_aligned = scale * (gen_lm @ R.T) + (sc - scale * (R @ gc))

            nme = float(np.linalg.norm(src_lm - gen_aligned, axis=1).mean()) / iod
            nmes.append(nme)

    if not nmes:
        return {"error": "no valid landmark pairs found"}

    a = np.array(nmes)
    return {
        "mean_nme":       round(float(np.mean(a)), 5),
        "p95_nme":        round(float(np.percentile(a, 95)), 5),
        "no_face_frames": no_face,
        "n_frames":       len(nmes),
        "note": "Lower NME = better expression/pose transfer. Normalised by inter-ocular distance.",
    }


# ── METRIC 3: Temporal Stability — Temporal LPIPS ────────────────────────────

def metric_temporal_lpips(gen_frames):
    """
    LPIPS distance between consecutive aligned face crop frames.
    Mean and p95 over all consecutive pairs.
    """
    if not HAS_TORCH:
        return {"error": "pip install torch lpips"}
    if not HAS_LPIPS:
        return {"error": "pip install lpips"}
    if not HAS_TORCHVISION:
        return {"error": "pip install torchvision"}

    if not HAS_INSIGHTFACE:
        crops = [r(f, (256, 256)) for f in gen_frames]
    else:
        with suppress_c_output():
            app = FaceAnalysis(name="buffalo_l", providers=ONNX_PROVIDERS)
            app.prepare(ctx_id=INSIGHTFACE_CTX, det_size=(640, 640))
        crops = []
        for frame in gen_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = app.get(rgb)
            if faces:
                b = faces[0].bbox.astype(int)
                x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), b[2], b[3]
                crop = frame[y1:y2, x1:x2]
                crops.append(r(crop, (256, 256)) if crop.size > 0 else r(frame, (256, 256)))
            else:
                crops.append(r(frame, (256, 256)))

    loss_fn = _lpips.LPIPS(net="alex").to(DEVICE).eval()
    tf = _T.Compose([_T.ToTensor(), _T.Normalize([0.5]*3, [0.5]*3)])

    def to_t(f):
        return tf(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)

    dists = []
    for i in range(len(crops) - 1):
        with torch.no_grad():
            dists.append(float(loss_fn(to_t(crops[i]), to_t(crops[i+1])).item()))

    if not dists:
        return {"error": "not enough frames"}

    a = np.array(dists)
    return {
        "mean":    round(float(np.mean(a)), 4),
        "p95":     round(float(np.percentile(a, 95)), 4),
        "n_pairs": len(a),
        "note": "Lower = more temporally stable. p95 captures worst flickering.",
    }


# ── METRIC 4: Visual Realism — FID on face crops ─────────────────────────────

def metric_fid(src_frames, gen_frames):
    """
    FID between InceptionV3 pool3 features of real and generated face crops.
    """
    if not HAS_TORCH:
        return {"error": "pip install torch torchvision scipy"}
    if not HAS_TORCHVISION:
        return {"error": "pip install torchvision"}
    if not HAS_SCIPY:
        return {"error": "pip install scipy"}
    if not src_frames:
        return {"error": "provide --source for FID"}

    inception = _models.inception_v3(weights=_models.Inception_V3_Weights.DEFAULT)
    extractor = torch.nn.Sequential(
        inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
        torch.nn.MaxPool2d(3, 2), inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
        torch.nn.MaxPool2d(3, 2),
        inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
        inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
        inception.Mixed_6d, inception.Mixed_6e,
        inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
        torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(),
    ).to(DEVICE).eval()

    tf = _T.Compose([
        _T.ToTensor(),
        _T.Resize((299, 299), antialias=True),
        _T.Normalize([0.5]*3, [0.5]*3),
    ])

    def extract_feats(frames):
        if HAS_INSIGHTFACE:
            with suppress_c_output():
                app = FaceAnalysis(name="buffalo_l", providers=ONNX_PROVIDERS)
                app.prepare(ctx_id=INSIGHTFACE_CTX, det_size=(640, 640))
            crops = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = app.get(rgb)
                if faces:
                    b = faces[0].bbox.astype(int)
                    x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), b[2], b[3]
                    crop = frame[y1:y2, x1:x2]
                    crops.append(crop if crop.size > 0 else frame)
                else:
                    crops.append(frame)
        else:
            crops = frames

        tensors = [tf(cv2.cvtColor(cv2.resize(c, (299, 299)), cv2.COLOR_BGR2RGB))
                   for c in crops]
        feats = []
        for i in range(0, len(tensors), 32):
            batch = torch.stack(tensors[i:i+32]).to(DEVICE)
            with torch.no_grad():
                feats.append(extractor(batch).cpu().numpy())
        return np.concatenate(feats)

    real_feats = extract_feats(src_frames)
    gen_feats  = extract_feats(gen_frames)

    mu_r, s_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, s_g = gen_feats.mean(0),  np.cov(gen_feats,  rowvar=False)
    diff = mu_r - mu_g
    cov, _ = _sqrtm(s_r @ s_g, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real

    fid = float(diff @ diff + np.trace(s_r + s_g - 2 * cov))
    return {
        "fid":    round(fid, 3),
        "n_real": len(real_feats),
        "n_gen":  len(gen_feats),
        "note": "Lower FID = more realistic. Computed on InceptionV3 pool3 face crop features.",
    }


# ── METRIC 5: Real-time Performance — Latency ────────────────────────────────

def metric_latency(perf_json_path=None, video_info=None):
    """
    Reads per-chunk latency from the .perf.json sidecar written by
    infer_with_timing.py.  Computes p95 from the full chunk_latencies_ms list.

    If no perf_json is provided, returns not_measured with instructions.
    """
    base = {}
    if video_info:
        base["video_fps"]        = video_info.get("fps")
        base["video_duration_s"] = video_info.get("duration_s")

    if perf_json_path is None or not Path(perf_json_path).exists():
        return {
            **base,
            "status": "not_measured",
            "note": (
                "Pass --perf_json path/to/clip.perf.json (written by infer_with_timing.py) "
                "to populate latency metrics."
            ),
        }

    try:
        with open(perf_json_path, "r") as f:
            perf = json.load(f)
    except Exception as e:
        return {**base, "status": "error", "note": f"Could not read perf.json: {e}"}

    latencies = perf.get("chunk_latencies_ms", [])
    if not latencies:
        return {**base, "status": "error", "note": "perf.json has no chunk_latencies_ms"}

    a = np.array(latencies)
    p95 = float(np.percentile(a, 95))
    return {
        **base,
        "status":              "measured",
        "p95_latency_ms":      round(p95, 2),
        "p95_fps":             round(1000.0 / p95, 2) if p95 > 0 else None,
        "mean_latency_ms":     round(float(np.mean(a)), 2),
        "min_latency_ms":      round(float(np.min(a)), 2),
        "max_latency_ms":      round(float(np.max(a)), 2),
        "total_latency_ms":    round(float(np.sum(a)), 2),
        "num_chunks":          len(latencies),
        "block_frame_ms":      perf.get("block_frame_ms"),
        "read_chunk_size":     perf.get("read_chunk_size"),
        "precision":           perf.get("precision"),
        "device":              perf.get("device"),
        "peak_gpu_memory_mb":  perf.get("peak_gpu_memory_mb"),
        "note": "p95 end-to-end chunk latency measured during RVC inference.",
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Group O1 — Video Quality Metrics")
    p.add_argument("--generated",  required=True,  help="Generated output video (.mp4)")
    p.add_argument("--reference",  required=True,  help="Reference avatar image (for ArcFace)")
    p.add_argument("--source",     default=None,   help="Original source video (for NME, FID)")
    p.add_argument("--perf_json",  default=None,   help="Path to .perf.json sidecar from infer_with_timing.py")
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--out",        default=None,   help="Output JSON path")
    args = p.parse_args()

    for path, name in [(args.generated, "--generated"), (args.reference, "--reference")]:
        if not Path(path).exists():
            print(f"ERROR: {name} not found: {path}"); sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Group O1 — Video Quality Metrics")
    print(f"{'='*60}")

    info = get_info(args.generated)
    print(f"\n  Generated: {args.generated}")
    print(f"  Size:      {info['width']}x{info['height']} @ {info['fps']} fps | {info['duration_s']}s")
    if args.source:
        print(f"  Source:    {args.source}")
    print(f"  Reference: {args.reference}")

    print(f"\n  Loading frames (max {args.max_frames})...")
    gen_frames, _ = read_frames(args.generated, args.max_frames)
    src_frames = None
    if args.source and Path(args.source).exists():
        src_frames, _ = read_frames(args.source, args.max_frames)
        n = min(len(src_frames), len(gen_frames))
        src_frames, gen_frames = src_frames[:n], gen_frames[:n]

    ref_image = cv2.imread(args.reference)
    if ref_image is None:
        print(f"ERROR: could not read reference image: {args.reference}"); sys.exit(1)

    results = {}
    t0_total = time.perf_counter()

    print(f"\n  [1/5] Identity Preservation (ArcFace)...", end=" ", flush=True)
    results["identity_arcface"] = metric_arcface(ref_image, gen_frames)
    print(f"mean={results['identity_arcface'].get('mean', '?')}")

    print(f"  [2/5] Landmark NME (MediaPipe)...", end=" ", flush=True)
    results["landmark_nme"] = metric_landmark_nme(src_frames or [], gen_frames)
    print(f"mean_nme={results['landmark_nme'].get('mean_nme', '?')}")

    print(f"  [3/5] Temporal LPIPS...", end=" ", flush=True)
    results["temporal_lpips"] = metric_temporal_lpips(gen_frames)
    print(f"mean={results['temporal_lpips'].get('mean', '?')}")

    print(f"  [4/5] FID (InceptionV3 face crops)...", end=" ", flush=True)
    results["fid"] = metric_fid(src_frames, gen_frames)
    print(f"fid={results['fid'].get('fid', '?')}")

    print(f"  [5/5] Latency...", end=" ", flush=True)
    results["latency"] = metric_latency(args.perf_json, info)
    status = results["latency"].get("status", "?")
    if status == "measured":
        print(f"p95={results['latency'].get('p95_latency_ms', '?')} ms")
    else:
        print(f"{status}")

    elapsed = time.perf_counter() - t0_total

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")

    def row(label, d, key, fmt=".4f", unit=""):
        if isinstance(d, dict) and key in d and not isinstance(d[key], str):
            print(f"  {label:<40} {d[key]:{fmt}}{unit}")
        elif isinstance(d, dict) and "error" in d:
            print(f"  {label:<40} [error: {d['error']}]")
        elif isinstance(d, dict) and "skipped" in d:
            print(f"  {label:<40} [skipped]")

    print(f"\n  Identity Preservation (ArcFace)")
    row("  Mean cosine similarity",     results["identity_arcface"], "mean")
    row("  Std",                        results["identity_arcface"], "std")
    row("  p5 (worst identity drops)",  results["identity_arcface"], "p5")

    print(f"\n  Expression & Pose Transfer (Landmark NME)")
    row("  Mean NME",  results["landmark_nme"], "mean_nme", ".5f")
    row("  p95 NME",   results["landmark_nme"], "p95_nme",  ".5f")

    print(f"\n  Temporal Stability (LPIPS)")
    row("  Mean LPIPS", results["temporal_lpips"], "mean")
    row("  p95 LPIPS",  results["temporal_lpips"], "p95")

    print(f"\n  Visual Realism (FID)")
    row("  FID", results["fid"], "fid", ".3f")

    print(f"\n  Real-time Performance")
    lat = results.get("latency", {})
    if lat.get("status") == "measured":
        row("  p95 latency",   lat, "p95_latency_ms", ".1f", " ms")
        row("  p95 FPS",       lat, "p95_fps",        ".1f", " fps")
        row("  Mean latency",  lat, "mean_latency_ms", ".1f", " ms")
    else:
        print(f"  {'Latency':<40} [not measured — pass --perf_json]")

    print(f"\n  Total compute time: {elapsed:.1f}s")
    print(f"{'='*60}")

    metrics_dir = Path(__file__).parent / "metrics_results"
    metrics_dir.mkdir(exist_ok=True)
    default_name = Path(args.generated).stem + ".metrics.json"
    out_path = args.out or str(metrics_dir / default_name)
    with open(out_path, "w") as f:
        json.dump({"video": args.generated, "info": info, "metrics": results}, f, indent=2)
    print(f"\n  Saved: {out_path}\n")


if __name__ == "__main__":
    main()