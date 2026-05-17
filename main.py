"""
main.py - Property Classifier Pipeline
Usage:
  python main.py                  full pipeline + server
  python main.py --train-only     train only
  python main.py --serve          serve existing model
  python main.py --no-finetune    skip fine-tuning
  python main.py --skip-mit       skip MIT download (quick test)
  python main.py --local-data P   use local images
"""
import os, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"]      = "utf-8"

import argparse, time
from pathlib import Path
from logger import get_logger
from exception import AppException

log = get_logger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Property Classifier")
    p.add_argument("--serve",       action="store_true")
    p.add_argument("--train-only",  action="store_true")
    p.add_argument("--no-finetune", action="store_true")
    p.add_argument("--skip-mit",    action="store_true")
    p.add_argument("--local-data",  type=str, default=None)
    return p.parse_args()


def train(args):
    t0 = time.time()
    log.info("=" * 55)
    log.info("  Property Classifier  -  Pipeline Start")
    log.info("=" * 55)

    from src.data_ingestion import DataIngestion
    splits = DataIngestion(local_dir=args.local_data,
                           skip_mit=args.skip_mit).run()

    from src.data_transformation import DataTransformation
    tr = DataTransformation(train_dir=splits["train"],
                            val_dir=splits["val"],
                            test_dir=splits["test"])
    train_ds, val_ds, test_ds, class_names = tr.run()
    aug = tr.get_augmentation_layer()
    log.info(f"Dataset: {len(class_names)} classes | 224x224 | batch=32")

    from src.model_trainer import ModelTrainer
    trainer = ModelTrainer(num_classes=len(class_names),
                           augmentation_layer=aug,
                           class_names=class_names)
    model = trainer.run(train_ds=train_ds, val_ds=val_ds,
                        fine_tune=not args.no_finetune)

    log.info("-" * 55)
    log.info("  Test Set Evaluation")
    log.info("-" * 55)
    loss, acc = model.evaluate(test_ds, verbose=1)
    log.info(f"Test loss={loss:.4f} | accuracy={acc:.4f} ({acc*100:.1f}%)")

    ckpts = sorted(Path("models/checkpoints").glob("epoch_*.keras"))
    log.info(f"Checkpoints: {len(ckpts)} saved")
    log.info(f"Total time : {time.time()-t0:.1f}s")
    return model


def serve():
    log.info("Starting Property Classifier at http://localhost:8000")
    log.info("Prometheus metrics: http://localhost:8000/metrics")
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000,
                reload=False, log_level="error")


if __name__ == "__main__":
    args = parse_args()
    try:
        model_exists = Path("models/building_classifier.keras").exists()
        if args.serve and model_exists:
            log.info("Model found. Starting server.")
            serve()
        elif args.train_only:
            train(args)
        else:
            if not model_exists:
                train(args)
            else:
                log.info("Model already trained. Use --train-only to retrain.")
            serve()
    except AppException as e:
        log.error(f"Pipeline failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Stopped.")
        sys.exit(0)
