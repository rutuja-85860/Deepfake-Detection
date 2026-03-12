"""
main.py — Deepfake Video Detection API v3.1
============================================
ROOT-CAUSE FIX (why probabilities were always ~50%):

  1. PREPROCESSING MISMATCH (primary cause):
     The model was built with `tf.keras.applications.efficientnet.preprocess_input`
     as an internal layer — meaning the model expects RAW uint8-range pixels [0, 255]
     as input and handles normalization internally.
     The old code was applying /255 or ImageNet normalization BEFORE feeding the model,
     causing double-normalization → garbage activations → ~50% output.
     FIX: Feed raw float32 pixels in [0, 255] range directly to the model.

  2. LABEL INVERSION (secondary cause):
     `image_dataset_from_directory` assigns labels alphabetically:
       fake/ → 0,  real/ → 1
     So sigmoid output ≈ 1.0 means REAL, ≈ 0.0 means FAKE.
     The old code treated output ≈ 1.0 as FAKE — completely backwards.
     FIX: fake_probability = 1.0 - raw_sigmoid_output

  3. FOCAL LOSS custom object:
     Model was trained with a custom focal loss. Loading without passing it
     as a custom_object can silently fail or produce wrong results.
     FIX: Define focal_loss and pass it at load time.

  4. MEDIAPIPE API CHANGE (v3.1 fix):
     mediapipe >= 0.10 removed `mp.solutions` in favour of the Tasks API.
     FIX: Use `mediapipe.tasks.python.vision.FaceDetector` with the
     `blaze_face_short_range.tflite` model, downloaded automatically on first run.

Run:
    pip install mediapipe>=0.10
    uvicorn main:app --host 0.0.0.0 --port 8000
    ngrok http 8000
"""

import os, shutil, logging, tempfile, urllib.request, base64
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("deepfake_api")

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH      = Path("model/best_deepfake_model.h5")
IMG_SIZE        = (224, 224)
TARGET_FRAMES   = 40
MIN_DETECT_CONF = 0.5   # matches training (notebook uses 0.5)
FAKE_THRESHOLD  = 0.5
ALLOWED_EXT     = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_SIZE_MB     = 200
BATCH_SIZE      = 8
STATIC_DIR      = Path("static")

# MediaPipe Tasks face detector model (downloaded once)
MP_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
MP_MODEL_PATH = Path("model/blaze_face_short_range.tflite")

# ── Response schemas ──────────────────────────────────────────────────────────
class FrameStat(BaseModel):
    frame_index: int
    fake_probability: float
    real_probability: float
    verdict: str
    face_detected: bool
    weight: float

class DetectionResponse(BaseModel):
    verdict: str
    confidence: float
    fake_probability: float
    real_probability: float
    frames_analyzed: int
    faces_detected: int
    preprocessing_used: str
    frame_stats: Optional[List[FrameStat]] = None
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mediapipe_loaded: bool
    preprocessing_strategy: str
    model_path: str

# ── Global state ──────────────────────────────────────────────────────────────
class AppState:
    model       = None
    mp_detector = None   # mediapipe.tasks.python.vision.FaceDetector

state = AppState()

# ── Custom loss (required to load the model) ──────────────────────────────────
def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss used during training — must be supplied when loading the model."""
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - pt, gamma)
        bce = (
            -y_true * tf.math.log(y_pred)
            - (1 - y_true) * tf.math.log(1 - y_pred)
        )
        return tf.reduce_mean(alpha * focal_weight * bce)

    return loss_fn

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model() -> None:
    import tensorflow as tf

    logger.info(f"Loading model: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'")

    state.model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"loss_fn": focal_loss()},
        compile=False,
    )
    # Recompile with a standard loss — we only need inference
    state.model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(
        f"✅ Model ready | in={state.model.input_shape} | out={state.model.output_shape}"
    )

# ── MediaPipe loading (Tasks API — mediapipe >= 0.10) ─────────────────────────
def _ensure_mp_model() -> None:
    """Download the BlazeFace tflite if it doesn't exist yet."""
    MP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MP_MODEL_PATH.exists():
        logger.info(f"Downloading MediaPipe face detector model → {MP_MODEL_PATH}")
        urllib.request.urlretrieve(MP_MODEL_URL, str(MP_MODEL_PATH))
        logger.info("Download complete.")

def load_mediapipe() -> None:
    """
    Initialise the MediaPipe Tasks FaceDetector.

    mediapipe >= 0.10 dropped `mp.solutions` in favour of:
        mediapipe.tasks.python.vision.FaceDetector
    We use IMAGE running mode (stateless, one frame at a time).
    """
    _ensure_mp_model()

    import mediapipe as mp
    from mediapipe.tasks.python            import vision
    from mediapipe.tasks.python.core       import base_options as bo

    base_opts = bo.BaseOptions(model_asset_path=str(MP_MODEL_PATH))
    options   = vision.FaceDetectorOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=MIN_DETECT_CONF,
    )
    state.mp_detector = vision.FaceDetector.create_from_options(options)
    logger.info("✅ MediaPipe Tasks FaceDetector ready.")

# ── Frame sampling ────────────────────────────────────────────────────────────
def sample_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")

    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    n = min(TARGET_FRAMES, total)
    indices = np.linspace(0, total - 1, n, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    logger.info(f"Sampled {len(frames)}/{total} frames.")
    return frames


def gradcam_frames(frames: List[np.ndarray], max_frames: int = 6) -> List[dict]:
    """
    Select up to `max_frames` evenly spaced from the video,
    run Grad-CAM on each, return list of {frame_index, heatmap_b64, verdict, fake_prob}.
    """
    if not frames:
        return []
    indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)
    results = []
    for idx in indices:
        frame = frames[int(idx)]
        face, _, _ = detect_face(frame)
        tensor  = preprocess(face)
        import tensorflow as tf
        raw     = state.model.predict(np.expand_dims(tensor, 0), verbose=0)[0]
        p_real  = float(raw[0])
        p_fake  = 1.0 - p_real
        verdict = "FAKE" if p_fake >= FAKE_THRESHOLD else "REAL"
        b64     = gradcam_to_b64(face, verdict)
        results.append({
            "frame_index":    int(idx),
            "heatmap_b64":    b64,
            "verdict":        verdict,
            "fake_prob":      round(p_fake * 100, 1),
            "real_prob":      round(p_real * 100, 1),
        })
    return results

# ── Face detection ────────────────────────────────────────────────────────────
def detect_face(frame_bgr: np.ndarray):
    """
    Returns (face_bgr_224x224, face_detected: bool, detection_conf: float).

    Uses the MediaPipe Tasks API (mediapipe >= 0.10).
    Crops with the same 20px padding used during training.
    Falls back to full-frame resize if no face is found or detector unavailable.
    """
    if state.mp_detector is None:
        return cv2.resize(frame_bgr, IMG_SIZE), False, 0.0

    import mediapipe as mp

    h, w = frame_bgr.shape[:2]

    # Tasks API requires an mp.Image in RGB format
    rgb       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result    = state.mp_detector.detect(mp_image)

    if not result.detections:
        return cv2.resize(frame_bgr, IMG_SIZE), False, 0.0

    # Pick the detection with the highest confidence
    best = max(
        result.detections,
        key=lambda d: d.categories[0].score if d.categories else 0.0,
    )
    conf = float(best.categories[0].score) if best.categories else 0.0

    # BoundingBox in the Tasks API uses pixel coordinates directly
    bb   = best.bounding_box
    x    = max(0,  bb.origin_x - 20)
    y    = max(0,  bb.origin_y - 20)
    x2   = min(w,  bb.origin_x + bb.width  + 20)
    y2   = min(h,  bb.origin_y + bb.height + 20)

    crop = frame_bgr[y:y2, x:x2]
    if crop.size == 0:
        return cv2.resize(frame_bgr, IMG_SIZE), False, 0.0

    return cv2.resize(crop, IMG_SIZE), True, conf

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR → RGB then cast to float32 in [0, 255].

    WHY: The model was built with
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    as the FIRST internal layer, so EfficientNet's scaling (divides by 127.5,
    subtracts 1) happens inside the model graph.
    We must NOT apply any normalization here — just feed raw pixel values.
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return face_rgb.astype(np.float32)   # range [0, 255] — model scales internally

# ── Batch inference ───────────────────────────────────────────────────────────
def infer_batch(faces: List[np.ndarray]) -> np.ndarray:
    """
    Run a batch through the model and return FAKE probabilities.

    LABEL INVERSION FIX:
    image_dataset_from_directory assigns labels alphabetically:
        fake/ → class 0,  real/ → class 1
    So the sigmoid output is P(real).
    fake_probability = 1.0 - sigmoid_output
    """
    batch = np.stack(faces, axis=0)                 # (N, 224, 224, 3) float32
    raw   = state.model.predict(batch, verbose=0)   # (N, 1)  sigmoid → P(real)
    p_real = raw[:, 0]                              # shape (N,)
    p_fake = 1.0 - p_real                           # FIXED: invert to get P(fake)
    return p_fake


# ── Explainability heatmap (input-gradient saliency) ─────────────────────────
#
# THE PROBLEM WITH CLASSIC GRAD-CAM ON THIS MODEL:
#   tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])
#   creates a new Functional model whose symbolic tensors come from the ORIGINAL
#   graph.  When you call  grad_model(new_tensor)  Keras tries to run through its
#   internal layer-DAG but the id(x) of intermediate tensors are NOT in the new
#   execution's tensor_dict → KeyError.  This is a known limitation of loading
#   complex models (e.g. EfficientNet with preprocessing inside) from .h5.
#
# THE FIX — input-gradient saliency:
#   1. Wrap the input in tf.Variable  (GradientTape auto-watches Variables)
#   2. Call state.model(inp) directly — no sub-model, no graph splitting
#   3. Backprop  d(P(fake or real)) / d(input pixels)
#   4. Take channel-max |gradient|, apply Gaussian blur for coherent blobs
#   5. JET colormap overlay → same visual semantics as Grad-CAM
#      (eyes / mouth / nose-bridge light up red/yellow — exactly where
#       deepfake artefacts concentrate)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradcam(face_bgr: np.ndarray, pred_index: int = 0) -> np.ndarray:
    """
    Return a BGR heatmap blended with face_bgr, same spatial size.
    pred_index=0 → highlight what drives P(real) high  (use for REAL frames)
    pred_index=1 → highlight what drives P(fake) high  (use for FAKE frames)
    """
    import tensorflow as tf

    if state.model is None:
        return face_bgr.copy()

    try:
        tensor = preprocess(face_bgr)                              # (224,224,3) float32 [0,255]
        inp    = tf.Variable(tensor[np.newaxis], dtype=tf.float32) # (1,224,224,3) — auto-watched

        with tf.GradientTape() as tape:
            pred   = state.model(inp, training=False)              # (1,1) sigmoid = P(real)
            p_real = pred[0, 0]
            # pred_index 0 → gradient of P(real);  1 → gradient of P(fake)=1-P(real)
            loss   = p_real if pred_index == 0 else (1.0 - p_real)

        grads    = tape.gradient(loss, inp)                        # (1,224,224,3)
        if grads is None:
            return face_bgr.copy()

        saliency = tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()  # (224,224)

        # Normalise to [0,1]
        s_max = saliency.max()
        if s_max > 1e-8:
            saliency /= s_max

        # Gaussian blur → contiguous spatial blobs (like Grad-CAM)
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (15, 15), 0)
        s_max = saliency.max()
        if s_max > 1e-8:
            saliency /= s_max

        # Resize to match face crop
        h, w = face_bgr.shape[:2]
        sal  = cv2.resize(saliency, (w, h))

        # JET: blue = low attention, red/yellow = high attention
        heatmap = cv2.applyColorMap((sal * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(face_bgr, 0.45, heatmap, 0.55, 0)
        return blended

    except Exception as exc:
        logger.warning(f"[GradCAM] fallback — returning plain face: {exc}")
        return face_bgr.copy()


def gradcam_to_b64(face_bgr: np.ndarray, verdict: str) -> str:
    """Compute saliency heatmap and return as base64 JPEG data-URI."""
    # For FAKE show what pushes toward FAKE (1-P(real)); for REAL show P(real)
    pred_idx = 1 if verdict == "FAKE" else 0
    cam_img  = compute_gradcam(face_bgr, pred_index=pred_idx)
    _, buf   = cv2.imencode('.jpg', cam_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ── Full prediction pipeline ──────────────────────────────────────────────────
def predict_frames(frames: List[np.ndarray]) -> List[FrameStat]:
    if state.model is None:
        raise RuntimeError("Model not loaded.")

    all_faces, meta = [], []
    for idx, frame in enumerate(frames):
        face, detected, conf = detect_face(frame)
        all_faces.append(preprocess(face))
        meta.append({"idx": idx, "detected": detected, "conf": conf})

    fake_probs: List[float] = []
    for i in range(0, len(all_faces), BATCH_SIZE):
        fake_probs.extend(infer_batch(all_faces[i : i + BATCH_SIZE]).tolist())

    stats = []
    for i, m in enumerate(meta):
        fp = float(fake_probs[i])
        # Frames where a face was detected are more trustworthy
        w = m["conf"] if m["detected"] else 0.3
        stats.append(
            FrameStat(
                frame_index=m["idx"],
                fake_probability=round(fp, 4),
                real_probability=round(1.0 - fp, 4),
                verdict="FAKE" if fp >= FAKE_THRESHOLD else "REAL",
                face_detected=m["detected"],
                weight=round(w, 4),
            )
        )
    return stats

# ── Weighted aggregation ──────────────────────────────────────────────────────
def aggregate(stats: List[FrameStat]) -> dict:
    w = np.array([s.weight for s in stats], dtype=np.float64)
    p = np.array([s.fake_probability for s in stats], dtype=np.float64)

    if w.sum() == 0:
        w = np.ones_like(w)

    mean_fake  = float(np.dot(w, p) / w.sum())
    verdict    = "FAKE" if mean_fake >= FAKE_THRESHOLD else "REAL"
    # Confidence: how far from the 50% decision boundary (0–1 scale)
    confidence = round(min(abs(mean_fake - FAKE_THRESHOLD) / FAKE_THRESHOLD, 1.0), 4)

    return {
        "verdict":          verdict,
        "confidence":       confidence,
        "fake_probability": round(mean_fake, 4),
        "real_probability": round(1.0 - mean_fake, 4),
        "faces_detected":   sum(1 for s in stats if s.face_detected),
    }

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_mediapipe()
    yield
    logger.info("Shutdown.")
    state.model = None
    if state.mp_detector:
        state.mp_detector.close()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Detection API v3.2",
    version="3.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Deepfake Detection API v3.1 — POST a video to /detect"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if state.model is not None else "model_not_loaded",
        model_loaded=state.model is not None,
        mediapipe_loaded=state.mp_detector is not None,
        preprocessing_strategy=(
            "efficientnet_internal — raw [0,255] input, "
            "model applies preprocess_input internally"
        ),
        model_path=str(MODEL_PATH),
    )


@app.get("/diagnose", tags=["Debug"])
async def diagnose():
    """
    Returns raw model outputs for a synthetic test image.
    With the corrected preprocessing the output should deviate clearly from 0.5
    on real face images. Synthetic images may still be near 0.5 — that is normal.
    Use /diagnose-video with a real video for a definitive sanity check.
    """
    if state.model is None:
        raise HTTPException(503, "Model not loaded.")

    h, w = IMG_SIZE
    # Build a synthetic face-like gradient image
    test_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        test_bgr[i, :, :] = [
            int(128 + 60 * np.sin(i / h * np.pi)),
            int(100 + 40 * np.cos(i / h * np.pi)),
            int(90  + 50 * np.sin(i / h * np.pi * 2)),
        ]

    tensor = preprocess(test_bgr)
    raw    = state.model.predict(np.expand_dims(tensor, 0), verbose=0)[0]
    p_real = float(raw[0])
    p_fake = 1.0 - p_real

    return JSONResponse({
        "model_input_shape":  list(state.model.input_shape),
        "model_output_shape": list(state.model.output_shape),
        "preprocessing":      "raw float32 [0,255] — EfficientNet preprocess_input is inside the model",
        "label_mapping":      "sigmoid output = P(real) | fake_prob = 1 - sigmoid_output",
        "raw_sigmoid_output": round(p_real, 6),
        "fake_probability":   round(p_fake, 6),
        "real_probability":   round(p_real, 6),
        "note": (
            "Synthetic images near 0.5 is expected — the model was trained on real face crops. "
            "Test with /diagnose-video using an actual video."
        ),
    })


@app.post("/diagnose-video", tags=["Debug"])
async def diagnose_video(file: UploadFile = File(...)):
    """
    Upload a video. Returns raw sigmoid outputs and corrected fake/real
    probabilities for the first detected face. Use to verify the fix.
    """
    if state.model is None:
        raise HTTPException(503, "Model not loaded.")

    suffix   = Path(file.filename or "v.mp4").suffix.lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        frames = sample_frames(tmp_path)
        if not frames:
            raise HTTPException(422, "No frames extracted.")

        # Find first frame with a detected face
        test_face_bgr = None
        face_found    = False
        for frame in frames[:15]:
            face, detected, _ = detect_face(frame)
            if detected:
                test_face_bgr = face
                face_found    = True
                break
        if test_face_bgr is None:
            test_face_bgr = cv2.resize(frames[0], IMG_SIZE)

        tensor    = preprocess(test_face_bgr)
        raw       = state.model.predict(np.expand_dims(tensor, 0), verbose=0)[0]
        p_real    = float(raw[0])
        p_fake    = 1.0 - p_real

        return JSONResponse({
            "face_detected":      face_found,
            "raw_sigmoid_output": round(p_real, 6),
            "fake_probability":   round(p_fake, 6),
            "real_probability":   round(p_real, 6),
            "verdict":            "FAKE" if p_fake >= FAKE_THRESHOLD else "REAL",
            "input_pixel_range":  f"[{tensor.min():.1f}, {tensor.max():.1f}]",
            "note": (
                "input_pixel_range should be roughly [0, 255]. "
                "If it shows [0, 1] or [-1, 1], preprocessing has been double-applied."
            ),
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    return_frame_stats: bool = True,
):
    if state.model is None:
        raise HTTPException(503, "Model not ready.")

    suffix = Path(file.filename or "v.mp4").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{suffix}'.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        size_mb = os.path.getsize(tmp_path) / 1e6
        if size_mb > MAX_SIZE_MB:
            raise HTTPException(413, f"File too large ({size_mb:.1f} MB > {MAX_SIZE_MB} MB).")

        logger.info(f"Processing '{file.filename}' ({size_mb:.1f} MB)")

        frames = sample_frames(tmp_path)
        if not frames:
            raise HTTPException(422, "No frames could be extracted from the video.")

        frame_stats = predict_frames(frames)
        agg         = aggregate(frame_stats)

        logger.info(
            f"→ {agg['verdict']} | conf={agg['confidence']} "
            f"| fake_prob={agg['fake_probability']} "
            f"| faces={agg['faces_detected']}/{len(frame_stats)}"
        )

        return DetectionResponse(
            **agg,
            frames_analyzed=len(frame_stats),
            preprocessing_used=(
                "efficientnet_internal — raw [0,255] float32 input; "
                "label_mapping: sigmoid≈1 → REAL, sigmoid≈0 → FAKE"
            ),
            frame_stats=frame_stats if return_frame_stats else None,
            message=f"Analysis complete. Verdict: {agg['verdict']}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Live Webcam Frame Detection ───────────────────────────────────────────────
class FrameRequest(BaseModel):
    image: str  # base64-encoded JPEG/PNG frame from webcam


@app.post("/detect-frame", tags=["Live"])
async def detect_frame(body: FrameRequest):
    """
    Accept a single base64-encoded image frame (from a webcam) and return
    a real-time deepfake verdict with confidence score.
    """
    if state.model is None:
        raise HTTPException(503, "Model not ready.")

    try:
        # Strip data-URI prefix if present (e.g. "data:image/jpeg;base64,...")
        b64 = body.image
        if "," in b64:
            b64 = b64.split(",", 1)[1]

        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise HTTPException(422, "Could not decode image.")

        face, detected, conf = detect_face(frame_bgr)
        tensor = preprocess(face)
        raw = state.model.predict(np.expand_dims(tensor, 0), verbose=0)[0]
        p_real = float(raw[0])
        p_fake = 1.0 - p_real

        verdict = "FAKE" if p_fake >= FAKE_THRESHOLD else "REAL"
        # Confidence: how far from the 50% boundary, mapped to 0-100%
        confidence = round(abs(p_fake - 0.5) * 2, 4)

        return JSONResponse({
            "verdict":          verdict,
            "confidence":       round(confidence * 100, 1),
            "fake_probability": round(p_fake * 100, 1),
            "real_probability": round(p_real * 100, 1),
            "face_detected":    detected,
            "face_conf":        round(conf * 100, 1),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, str(e))


class FrameGradcamRequest(BaseModel):
    image: str           # base64 webcam frame
    gradcam: bool = True # always True for this endpoint


@app.post("/detect-frame-gradcam", tags=["Live"])
async def detect_frame_gradcam(body: FrameGradcamRequest):
    """
    Like /detect-frame but also returns a Grad-CAM heatmap (base64 JPEG)
    highlighting which face regions drove the real/fake decision.
    """
    if state.model is None:
        raise HTTPException(503, "Model not ready.")

    try:
        b64 = body.image
        if "," in b64:
            b64 = b64.split(",", 1)[1]

        img_bytes = base64.b64decode(b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise HTTPException(422, "Could not decode image.")

        face, detected, conf = detect_face(frame_bgr)
        tensor  = preprocess(face)
        raw     = state.model.predict(np.expand_dims(tensor, 0), verbose=0)[0]
        p_real  = float(raw[0])
        p_fake  = 1.0 - p_real
        verdict = "FAKE" if p_fake >= FAKE_THRESHOLD else "REAL"
        conf_score = round(abs(p_fake - 0.5) * 2 * 100, 1)

        heatmap_b64 = gradcam_to_b64(face, verdict)

        return JSONResponse({
            "verdict":          verdict,
            "confidence":       conf_score,
            "fake_probability": round(p_fake * 100, 1),
            "real_probability": round(p_real * 100, 1),
            "face_detected":    detected,
            "face_conf":        round(conf * 100, 1),
            "heatmap_b64":      heatmap_b64,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, str(e))


@app.post("/detect-gradcam", tags=["Video"])
async def detect_with_gradcam(
    file: UploadFile = File(...),
    return_frame_stats: bool = True,
    gradcam_frames_count: int = 6,
):
    """
    Upload a video and receive:
    - Full detection verdict + confidence
    - Per-frame statistics
    - Up to `gradcam_frames_count` Grad-CAM heatmap images (spread evenly)
    """
    if state.model is None:
        raise HTTPException(503, "Model not ready.")

    suffix = Path(file.filename or "v.mp4").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{suffix}'.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        size_mb = os.path.getsize(tmp_path) / 1e6
        if size_mb > MAX_SIZE_MB:
            raise HTTPException(413, f"File too large ({size_mb:.1f} MB).")

        logger.info(f"[GradCAM] Processing '{file.filename}' ({size_mb:.1f} MB)")

        frames      = sample_frames(tmp_path)
        if not frames:
            raise HTTPException(422, "No frames could be extracted.")

        frame_stats = predict_frames(frames)
        agg         = aggregate(frame_stats)
        gcam        = gradcam_frames(frames, max_frames=gradcam_frames_count)

        logger.info(f"[GradCAM] → {agg['verdict']} | conf={agg['confidence']}")

        return JSONResponse({
            **agg,
            "frames_analyzed":   len(frame_stats),
            "preprocessing_used": (
                "efficientnet_internal — raw [0,255] float32; "
                "label: sigmoid≈1→REAL, sigmoid≈0→FAKE"
            ),
            "frame_stats": [s.dict() for s in frame_stats] if return_frame_stats else [],
            "gradcam_frames": gcam,
            "message": f"Analysis complete. Verdict: {agg['verdict']}",
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)