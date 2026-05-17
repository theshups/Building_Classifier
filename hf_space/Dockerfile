# ── Base: Python 3.10 on slim Debian ──────────────────────────────────────
# python:3.10-slim is critical:
#   - Python 3.13 (HF default) removed audioop → gradio/pydub crash
#   - Python 3.10 has audioop built-in and is TF 2.15 compatible
FROM python:3.10-slim

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────
# libgomp1: required by TensorFlow for OpenMP multi-threading
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────
# numpy MUST be installed before tensorflow — TF checks numpy at install time
# Each on separate RUN so Docker caches each layer independently
# If you change app.py later, packages won't reinstall (fast rebuild)

RUN pip install --no-cache-dir "numpy==1.26.4"

# tensorflow-cpu: CPU-only build (~500MB) — HF Spaces has no GPU
RUN pip install --no-cache-dir "tensorflow-cpu==2.15.0"

# FastAPI stack — lightweight web server, no gradio dependency hell
RUN pip install --no-cache-dir \
    "fastapi==0.110.0" \
    "uvicorn[standard]==0.29.0" \
    "python-multipart==0.0.9" \
    "pillow==10.3.0" \
    "h5py==3.11.0"

# ── Copy application files ─────────────────────────────────────────────────
COPY app.py .

# Copy model files (must exist before building image or uploading to HF)
COPY models/ ./models/

# ── Runtime config ─────────────────────────────────────────────────────────
# Suppress TensorFlow C++ verbose logs
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0

# HF Spaces routes external traffic to this port
EXPOSE 7860

# ── Start command ──────────────────────────────────────────────────────────
# uvicorn serves FastAPI directly — no gradio server check, no localhost issues
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
