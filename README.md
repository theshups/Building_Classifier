# Building Classifier

AI-powered building environment classifier using ResNet101V2 transfer learning.

![CI Pipeline](https://github.com/YOUR_USERNAME/Building_Classifier/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/YOUR_USERNAME/Building_Classifier/actions/workflows/docker.yml/badge.svg)
![HuggingFace Deploy](https://github.com/YOUR_USERNAME/Building_Classifier/actions/workflows/deploy_hf.yml/badge.svg)

## Live Demo
**HuggingFace Space:** https://huggingface.co/spaces/balkotjokes/property-classifier

## Classes
| Class | Dataset Source |
|-------|---------------|
| Exterior Facade | CMP Facade DB (Czech Technical University) |
| Office Interior | MIT Indoor Scenes (MIT CSAIL) |
| Warehouse | MIT Indoor Scenes (MIT CSAIL) |
| HVAC Pipeline | Wikimedia Commons + Open Images V7 |

## Model Architecture
```
Input (224x224x3)
  -> Augmentation (RandomFlip, Rotation, Zoom, Contrast)
  -> ResNet101V2 (ImageNet, 380 layers, frozen)
  -> GlobalAveragePooling2D
  -> Dense(12, relu)         24,588 params
  -> Dropout(0.50)
  -> Dense(4, softmax)           52 params
                            ─────────────
  Trainable params:         24,640  (under 25K budget)
  Total params:         45,700,200
```

## Project Structure
```
Building_Classifier/
├── .github/
│   └── workflows/
│       ├── ci.yml           CI - lint and syntax check
│       ├── deploy_hf.yml    CD - deploy to HuggingFace Spaces
│       └── docker.yml       CD - build and push Docker image
├── src/
│   ├── data_ingestion.py    Download datasets (no API key)
│   ├── data_transformation.py  tf.data pipeline
│   ├── model_trainer.py     ResNet101V2 transfer learning
│   └── app.py               FastAPI inference server
├── hf_space/
│   ├── app.py               HuggingFace deployment app
│   ├── Dockerfile           HF Space container
│   └── README.md            HF Space metadata
├── monitoring/
│   ├── prometheus.yml       Prometheus scrape config
│   └── grafana/             Grafana dashboard + datasource
├── static/
│   └── index.html           Web UI
├── models/
│   ├── tokenizer.json       Class map + preprocessing config
│   └── class_names.json     Class labels
├── logger.py
├── exception.py
├── main.py                  Pipeline orchestrator
├── requirements.txt
├── Dockerfile               Local Docker image
├── docker-compose.yml       Full stack (app + prometheus + grafana)
├── install.bat              Windows installer
└── setup_tracking.bat       WandB + MLflow setup
```

## Quick Start

```powershell
# 1. Install dependencies
.\install.bat

# 2. Run full pipeline (download data + train + serve)
python main.py

# 3. Open UI
Start-Process "http://localhost:8000"
```

## GitHub Secrets Required

Go to **GitHub repo → Settings → Secrets → Actions → New secret**

| Secret | Value | Used by |
|--------|-------|---------|
| `HF_TOKEN` | HuggingFace write token | deploy_hf.yml |
| `DOCKERHUB_USERNAME` | Docker Hub username | docker.yml |
| `DOCKERHUB_TOKEN` | Docker Hub access token | docker.yml |

## Experiment Tracking
- **WandB:** https://wandb.ai
- **MLflow:** `python -m mlflow server --port 5000` → http://localhost:5000

## Docker
```powershell
# Local
docker build -t building-classifier .
docker run -d -p 8000:8000 building-classifier

# Full stack with monitoring
docker-compose up -d
```

## Tech Stack
TensorFlow 2.15 | ResNet101V2 | FastAPI | Uvicorn | WandB | MLflow | Prometheus | Grafana | Docker
