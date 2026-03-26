"""
metrics.py — Group O1, BUas ADS&AI Y2C 2026
5 video quality metrics on synchronized input–output video pairs.

Usage:
    python metrics.py --generated output.mp4 --reference avatar.jpg
    python metrics.py --generated output.mp4 --reference avatar.jpg --source original.mp4
"""

import cv2, json, time, argparse, sys, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CUDA detection ────────────────────────────────────────────────────────────
# Automatically selects GPU providers if CUDA is available.
# If onnxruntime-gpu is installed but no CUDA found, falls back to CPU.
# To switch to CPU-only: replace onnxruntime-gpu with onnxruntime in pyproject.toml
#   uv pip uninstall onnxruntime-gpu && uv pip install onnxruntime==1.19.0

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

try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import torch
    import lpips as _lpips
    from torchvision import transforms as _T, models as _models
    from scipy.linalg import sqrtm as _sqrtm
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"


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
    """
    if not HAS_INSIGHTFACE:
        return {"error": "pip install insightface onnxruntime-gpu"}

    app = FaceAnalysis(name="buffalo_l", providers=ONNX_PROVIDERS)
    app.prepare(ctx_id=INSIGHTFACE_CTX, det_size=(640, 640))

    # Reference embedding
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
        cos_sim = float(np.dot(ref_emb, emb))  # both normalised → dot = cosine sim
        sims.append(cos_sim)

    if not sims:
        return {"error": "No faces detected in any output frame"}

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
    Uses MediaPipe FaceMesh (new Tasks API or legacy solutions API).
    NME normalised by inter-ocular distance.
    """
    if not HAS_MEDIAPIPE:
        return {"error": "pip install mediapipe"}
    if not src_frames:
        return {"error": "provide --source for NME"}

    LEFT_EYE  = 33
    RIGHT_EYE = 263
    NOSE_TIP  = 1

    # Detect which MediaPipe API is available
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

    with FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        for sf, gf in zip(src_frames, gen_frames):
            h, w = sf.shape[:2]
            src_lm = get_lm(sf, fm)
            gen_lm = get_lm(gf, fm)

            if src_lm is None or gen_lm is None:
                no_face += 1
                continue

            iod = np.linalg.norm(src_lm[LEFT_EYE] - src_lm[RIGHT_EYE])
            if iod < 1e-6:
                continue

            # Similarity transform alignment
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
    if not HAS_INSIGHTFACE:
        # Fallback: use full frame crops if InsightFace unavailable
        crops = [r(f, (256, 256)) for f in gen_frames]
    else:
        # Crop aligned face regions
        app = FaceAnalysis(name="buffalo_l",
                           providers=ONNX_PROVIDERS)
        app.prepare(ctx_id=INSIGHTFACE_CTX, det_size=(640, 640))
        crops = []
        for frame in gen_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = app.get(rgb)
            if faces:
                # Use bounding box crop
                b = faces[0].bbox.astype(int)
                x1, y1, x2, y2 = max(0,b[0]), max(0,b[1]), b[2], b[3]
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
        "mean": round(float(np.mean(a)), 4),
        "p95":  round(float(np.percentile(a, 95)), 4),
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
        # Get face crops if InsightFace available, else resize full frame
        if HAS_INSIGHTFACE:
            app = FaceAnalysis(name="buffalo_l",
                               providers=ONNX_PROVIDERS)
            app.prepare(ctx_id=INSIGHTFACE_CTX, det_size=(640, 640))
            crops = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = app.get(rgb)
                if faces:
                    b = faces[0].bbox.astype(int)
                    x1, y1, x2, y2 = max(0,b[0]), max(0,b[1]), b[2], b[3]
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
        "fid": round(fid, 3),
        "n_real": len(real_feats),
        "n_gen":  len(gen_feats),
        "note": "Lower FID = more realistic. Computed on InceptionV3 pool3 face crop features.",
    }


# ── METRIC 5: Real-time Performance — Latency & FPS ──────────────────────────

def metric_latency(video_info=None):
    """
    Latency CANNOT be measured from a pre-recorded video file.
    It must be measured by instrumenting the generation model directly.

    HOW TO MEASURE (add to your generation loop):

        import time, numpy as np
        latencies = []

        for frame in input_stream:
            t0 = time.perf_counter()
            output_frame = model.generate(frame)   # full end-to-end inference
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

        p95_latency_ms = np.percentile(latencies, 95)
        p95_fps        = 1000.0 / p95_latency_ms
        print(f"p95 latency: {p95_latency_ms:.1f} ms | p95 FPS: {p95_fps:.1f}")

    WHAT TO REPORT:
        - p95 end-to-end latency in milliseconds (camera capture → rendered output)
        - p95 FPS under continuous streaming conditions
    """
    result = {
        "status": "not_measured",
        "note": (
            "Latency must be measured by instrumenting the generation model directly. "
            "See metric_latency() docstring for measurement instructions."
        ),
    }
    if video_info:
        result["video_fps"]        = video_info.get("fps")
        result["video_duration_s"] = video_info.get("duration_s")
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Group O1 — Video Quality Metrics")
    p.add_argument("--generated",  required=True,  help="Generated output video (.mp4)")
    p.add_argument("--reference",  required=True,  help="Reference avatar image (for ArcFace)")
    p.add_argument("--source",     default=None,   help="Original source video (for NME, FID)")
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

    # 1. Identity preservation
    print(f"\n  [1/5] Identity Preservation (ArcFace)...", end=" ", flush=True)
    results["identity_arcface"] = metric_arcface(ref_image, gen_frames)
    print(f"mean={results['identity_arcface'].get('mean', '?')}")

    # 2. Expression & pose transfer
    print(f"  [2/5] Landmark NME (MediaPipe)...", end=" ", flush=True)
    results["landmark_nme"] = metric_landmark_nme(src_frames or [], gen_frames)
    print(f"mean_nme={results['landmark_nme'].get('mean_nme', '?')}")

    # 3. Temporal stability
    print(f"  [3/5] Temporal LPIPS...", end=" ", flush=True)
    results["temporal_lpips"] = metric_temporal_lpips(gen_frames)
    print(f"mean={results['temporal_lpips'].get('mean', '?')}")

    # 4. Visual realism
    print(f"  [4/5] FID (InceptionV3 face crops)...", end=" ", flush=True)
    results["fid"] = metric_fid(src_frames, gen_frames)
    print(f"fid={results['fid'].get('fid', '?')}")

    # 5. Latency
    print(f"  [5/5] Latency...", end=" ", flush=True)
    results["latency"] = metric_latency(info)
    print("not measured — see docstring")

    elapsed = time.perf_counter() - t0_total

    # ── Summary ───────────────────────────────────────────────────────────────
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
    if lat.get("status") == "not_measured":
        print(f"  {'Latency':<40} [not measured — instrument your model directly]")
    else:
        row("  p95 latency",   lat, "p95_latency_ms", ".1f", " ms")
        row("  p95 FPS",       lat, "p95_fps",        ".1f", " fps")

    print(f"\n  Total compute time: {elapsed:.1f}s")
    print(f"{'='*60}")

    # Save to metrics_results folder next to metrics.py
    metrics_dir = Path(__file__).parent / "metrics_results"
    metrics_dir.mkdir(exist_ok=True)
    default_name = Path(args.generated).stem + ".metrics.json"
    out_path = args.out or str(metrics_dir / default_name)
    with open(out_path, "w") as f:
        json.dump({"video": args.generated, "info": info, "metrics": results}, f, indent=2)
    print(f"\n  Saved: {out_path}\n")


if __name__ == "__main__":
    main()