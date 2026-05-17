"""
src/app.py - Property Classifier API
FastAPI + Prometheus metrics + advanced UI
"""
import io, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger import get_logger
from exception import AppException

log = get_logger(__name__)

MODEL_PATH     = Path("models/building_classifier.keras")
TOKENIZER_PATH = Path("models/tokenizer.json")
STATIC_DIR     = Path("static")
INDEX_HTML     = STATIC_DIR / "index.html"
TOP_K          = 4

# ── Prometheus metrics ────────────────────────────────────────────────
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

PRED_COUNTER   = Counter("property_predictions_total",
                          "Total predictions", ["class_name"])
REQUEST_COUNTER= Counter("property_requests_total",
                          "Total API requests", ["endpoint", "method"])
LATENCY_HIST   = Histogram("property_prediction_latency_seconds",
                            "Prediction latency in seconds",
                            buckets=[.05,.1,.25,.5,1,2.5,5])
CONFIDENCE_HIST= Histogram("property_prediction_confidence",
                            "Prediction confidence score",
                            buckets=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
MODEL_LOADED   = Gauge("property_model_loaded", "1 if model is loaded")
ACTIVE_REQUESTS= Gauge("property_active_requests", "Active requests")


# ── Singleton model service ───────────────────────────────────────────
class ModelService:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            o = super().__new__(cls)
            o._model, o._tokenizer = None, None
            cls._instance = o
        return cls._instance

    def load(self):
        import tensorflow as tf
        import tensorflow.data
        if self._model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model not found: {MODEL_PATH}. Run `python main.py` first.")
            log.info(f"Loading model from {MODEL_PATH} ...")
            self._model = tf.keras.models.load_model(str(MODEL_PATH))
            MODEL_LOADED.set(1)
            log.info("Model loaded.")
        if self._tokenizer is None:
            if not TOKENIZER_PATH.exists():
                raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")
            with open(TOKENIZER_PATH, encoding="utf-8") as f:
                self._tokenizer = json.load(f)
            log.info(f"Tokenizer: {self._tokenizer['num_classes']} classes")

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def predict(self, image_bytes: bytes) -> dict:
        tok      = self.tokenizer
        img_size = tuple(tok["input_size"])
        classes  = tok["class_names"]

        img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img  = img.resize(img_size, Image.BILINEAR)
        arr  = np.array(img, dtype=np.float32) / 255.0
        arr  = np.expand_dims(arr, 0)

        start = time.time()
        probs = self.model.predict(arr, verbose=0)[0]
        dur   = time.time() - start

        top  = np.argsort(probs)[::-1][:TOP_K]
        pred = classes[int(top[0])]
        conf = float(probs[top[0]])

        # Update Prometheus metrics
        PRED_COUNTER.labels(class_name=pred).inc()
        LATENCY_HIST.observe(dur)
        CONFIDENCE_HIST.observe(conf)

        return {
            "top_class":   pred,
            "confidence":  conf,
            "latency_ms":  round(dur * 1000, 2),
            "top_k": [
                {"class": classes[int(i)], "probability": float(probs[i])}
                for i in top
            ],
        }


# ── FastAPI app ───────────────────────────────────────────────────────
app     = FastAPI(title="Property Classifier API", version="2.0.0")
service = ModelService()

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def track_requests(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    REQUEST_COUNTER.labels(
        endpoint=request.url.path, method=request.method).inc()
    response = await call_next(request)
    ACTIVE_REQUESTS.dec()
    return response


@app.on_event("startup")
async def startup():
    try:
        service.load()
    except FileNotFoundError as e:
        log.warning(f"Startup load skipped: {e}")


@app.get("/", include_in_schema=False)
async def root():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return JSONResponse({"status": "ok", "app": "Property Classifier"})


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": service._model is not None}


@app.get("/classes")
async def classes():
    try:
        tok = service.tokenizer
        return {"num_classes": tok["num_classes"], "classes": tok["class_names"]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a JPEG/PNG image and receive property classification results."""
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(415, f"Unsupported type: {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        result = service.predict(data)
        log.info(f"Predicted: {result['top_class']} "
                 f"({result['confidence']:.4f}) in {result['latency_ms']}ms")
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats")
async def stats():
    """Quick stats for the UI dashboard."""
    return {
        "model_loaded": service._model is not None,
        "classes":      service.tokenizer.get("class_names", []) if service._tokenizer else [],
        "backbone":     service.tokenizer.get("backbone", "ResNet101V2") if service._tokenizer else "ResNet101V2",
        "trainable_params": service.tokenizer.get("head_trainable", 24640) if service._tokenizer else 24640,
    }
