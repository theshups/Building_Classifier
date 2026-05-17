# 🏢 Property Classifier

> AI-powered building environment analysis system using ResNet101V2 transfer learning — classifies real-world property images into 4 categories with 24,640 trainable parameters.

---

## Live Demo

**HuggingFace Space:** https://huggingface.co/spaces/balkotjokes/property-classifier

Upload any building photo — get instant AI classification with confidence scores.

---

## How It Works

The system is built in 4 phases that run automatically end-to-end:

### Phase 1 — Data Ingestion
Real building images are downloaded automatically from 3 public sources — no API key required:

| Class | Source | Images |
|-------|--------|--------|
| `exterior_facade` | CMP Facade DB — Czech Technical University | ~600 |
| `office_interior` | MIT Indoor Scenes — MIT CSAIL | ~600 |
| `warehouse` | MIT Indoor Scenes — MIT CSAIL | ~600 |
| `hvac_pipeline` | Wikimedia Commons + Google Open Images V7 | ~600 |

Images are split **70% train / 15% val / 15% test** and fed into a `tf.data` pipeline with `.cache()` and `.prefetch(AUTOTUNE)` so the GPU never waits on the CPU.

### Phase 2 — Transfer Learning
The model uses **ResNet101V2** pretrained on ImageNet as a frozen feature extractor. Only a tiny 3-layer head is trained — keeping trainable parameters well under the 25K budget:

```
Input Image (224 × 224 × 3)
       │
       ▼
Augmentation Layer        ← RandomFlip, Rotation ±12°, Zoom ±12%,
       │                     Contrast ±20%, Brightness ±15%
       │                     (GPU-side, disabled at inference)
       ▼
ResNet101V2 Backbone      ← 380 layers, 44,675,560 params
       │                     Pretrained on ImageNet, FROZEN
       ▼
GlobalAveragePooling2D    ← Compresses (7×7×2048) → 2048-d vector
       │
       ▼
Dense(12, relu)           ← 2048 × 12 + 12 = 24,588 params  ┐
       │                                                       │ trainable
Dropout(0.50)             ← Regularisation, disabled at test  │
       │                                                       │
Dense(4, softmax)         ← 12 × 4 + 4 = 52 params          ┘
       │
       ▼
Class Probabilities       ← [exterior_facade, office_interior,
                              warehouse, hvac_pipeline]

Total trainable: 24,640   (under 25K budget)
Total params:    45,700,200
```

Training uses **two phases**:
- **Phase A** — Backbone frozen, only head trains for 20 epochs (LR = 0.001)
- **Phase B** — Top 40 backbone layers unfreeze, fine-tune for 10 epochs (LR = 0.000005)

Callbacks used: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` (every epoch + best only), `WandbCallback`, `MLflowCallback`.

Class imbalance is handled with **automatic class weighting**.

### Phase 3 — Inference API
A **FastAPI** server wraps the trained model and exposes 3 endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Interactive web UI |
| `/predict` | POST | Upload image → predicted class + probabilities |
| `/health` | GET | Model loaded status |
| `/metrics` | GET | Prometheus metrics |

The model is loaded once at startup and reused across all requests — no reloading per request.

### Phase 4 — Monitoring
Every prediction is tracked by Prometheus and visualised in Grafana:
- Request rate and latency histogram
- Confidence score distribution
- Class prediction breakdown
- Active request count

---

## Project Structure

```
Building_Classifier/
│
├── .github/workflows/
│   ├── ci.yml              Lint + syntax check on every push
│   ├── deploy_hf.yml       Auto-deploy to HuggingFace on push to main
│   └── docker.yml          Build + push Docker image to Docker Hub
│
├── src/
│   ├── data_ingestion.py   Downloads datasets, splits 70/15/15
│   ├── data_transformation.py  tf.data pipeline (cache + prefetch)
│   ├── model_trainer.py    ResNet101V2 training + WandB + MLflow
│   └── app.py              FastAPI server + Prometheus metrics
│
├── hf_space/
│   ├── app.py              HuggingFace-specific FastAPI app
│   ├── Dockerfile          python:3.10-slim container for HF Spaces
│   └── README.md           HuggingFace Space metadata
│
├── monitoring/
│   ├── prometheus.yml      Scrape config (app:8000/metrics every 10s)
│   └── grafana/
│       ├── datasources/    Prometheus datasource config
│       └── dashboards/     Pre-built Grafana dashboard JSON
│
├── static/
│   └── index.html          Dark-themed drag-and-drop web UI
│
├── models/
│   ├── tokenizer.json      Class map + preprocessing config
│   └── class_names.json    Label list
│
├── logger.py               UTF-8 safe logging → logs/ folder
├── exception.py            Custom exception with file + line traceback
├── main.py                 Pipeline orchestrator (all 4 phases)
├── requirements.txt        Pinned Python dependencies
├── Dockerfile              Local Docker image
├── docker-compose.yml      App + Prometheus + Grafana + MLflow
├── install.bat             One-click Windows dependency installer
└── setup_tracking.bat      WandB login + MLflow server setup
```

---

## Quick Start

```powershell
# 1. Install all dependencies
.\install.bat

# 2. Setup experiment tracking (WandB + MLflow)
.\setup_tracking.bat

# 3. Run full pipeline
#    Downloads data -> trains model -> starts API server
python main.py

# 4. Open the web UI
Start-Process "http://localhost:8000"
```

### CLI Options

```powershell
python main.py --train-only      # train without starting server
python main.py --serve           # serve existing trained model
python main.py --no-finetune     # skip fine-tuning phase (faster)
python main.py --skip-mit        # skip 2.4GB MIT download (quick test)
python main.py --local-data PATH # use your own images
```

---

## Saved Artifacts

After training completes:

```
models/
  building_classifier.keras    final trained model
  tokenizer.json               class names + preprocessing config
  training_history.csv         loss/accuracy per epoch
  checkpoints/
    epoch_001.keras            snapshot every epoch
    best.keras                 best validation accuracy
```

---

## Experiment Tracking

**WandB** — real-time training curves:
```
https://wandb.ai
```

**MLflow** — local experiment dashboard:
```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```
Open: `http://localhost:5000` → login: `admin / property123`

---

## Docker

```powershell
# Build and run locally
docker build -t building-classifier .
docker run -d -p 8000:8000 --name bc building-classifier

# Full monitoring stack (App + Prometheus + Grafana + MLflow)
docker-compose up -d
```

| Service | URL | Login |
|---------|-----|-------|
| Property Classifier | http://localhost:8000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / property123 |
| MLflow | http://localhost:5000 | admin / property123 |

---

## GitHub Actions (CI/CD)

| Workflow | Triggers when | Does |
|----------|--------------|------|
| `ci.yml` | Any push | Checks Python syntax in all files |
| `deploy_hf.yml` | Push to `main` (hf_space/ changed) | Uploads to HuggingFace automatically |
| `docker.yml` | Push to `main` (Dockerfile changed) | Builds + pushes to Docker Hub |

### Required Secrets

Go to **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Where to get |
|--------|-------------|
| `HF_TOKEN` | https://huggingface.co/settings/tokens (Write access) |
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | https://app.docker.com/settings/personal-access-tokens |

---

## API Reference

```bash
# Health check
curl http://localhost:8000/health

# Classify an image
curl -X POST http://localhost:8000/predict -F "file=@building.jpg"

# Response
{
  "top_class": "Office Interior",
  "confidence": 0.8731,
  "latency_ms": 142.3,
  "predictions": [
    {"class": "Office Interior", "probability": 0.8731},
    {"class": "Exterior Facade", "probability": 0.0821},
    {"class": "Warehouse",       "probability": 0.0312},
    {"class": "HVAC Pipeline",   "probability": 0.0136}
  ]
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow 2.15, Keras, ResNet101V2 |
| Data Pipeline | tf.data, NumPy, Pillow |
| API Server | FastAPI, Uvicorn |
| Experiment Tracking | Weights & Biases, MLflow |
| Monitoring | Prometheus, Grafana |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | HuggingFace Spaces, Docker Hub |
| Language | Python 3.11 |
