"""
src/model_trainer.py
=====================
ResNet101V2 Transfer Learning  |  Target accuracy: > 60%

Architecture
------------
ResNet101V2 backbone  44,675,560  non-trainable (frozen Phase A)
Dense(256, relu) + L2    524,544  trainable
BatchNormalization         1,024  trainable
Dropout(0.40)                  0
Dense(128, relu) + L2     32,896  trainable
Dropout(0.30)                  0
Dense(4, softmax)            516  trainable
                         -------
Phase A trainable:       558,980

WandB : reads key from wandb_key.txt (accepts old 40-char and new wandb_v1_ format)
MLflow: local file tracking, no server or password needed
"""

import csv, json, os, sys, time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger import get_logger
from exception import AppException

log = get_logger(__name__)

MODELS_DIR     = Path("models")
CKPT_DIR       = MODELS_DIR / "checkpoints"
HISTORY_CSV    = MODELS_DIR / "training_history.csv"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"
MODEL_PATH     = MODELS_DIR / "building_classifier.keras"

IMG_SIZE         = (224, 224)
EPOCHS_HEAD      = 25
EPOCHS_FINETUNE  = 15
INITIAL_LR       = 1e-3
FINE_TUNE_LR     = 5e-6
FINE_TUNE_LAYERS = 60
L2_REG           = 1e-4
PROJECT_NAME     = "property-classifier"


# --------------------------------------------------------------------------- #
class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, phase, total):
        super().__init__()
        self.phase = phase
        self.total = total
        self._t    = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()
        log.info(f"[{self.phase}] Epoch {epoch+1}/{self.total} ...")

    def on_epoch_end(self, epoch, logs=None):
        d = logs or {}
        log.info(
            f"[{self.phase}] Epoch {epoch+1}/{self.total} "
            f"| loss={d.get('loss',0):.4f}  acc={d.get('accuracy',0):.4f} "
            f"| val_loss={d.get('val_loss',0):.4f}  val_acc={d.get('val_accuracy',0):.4f} "
            f"| {time.time()-self._t:.1f}s"
        )


# --------------------------------------------------------------------------- #
class WandbCallback(tf.keras.callbacks.Callback):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase

    def on_epoch_end(self, epoch, logs=None):
        try:
            import wandb
            if wandb.run:
                wandb.log(
                    {f"{self.phase}/{k}": v for k, v in (logs or {}).items()}
                    | {"epoch": epoch + 1}
                )
        except Exception:
            pass


# --------------------------------------------------------------------------- #
class MLflowCallback(tf.keras.callbacks.Callback):
    def __init__(self, phase, run_id):
        super().__init__()
        self.phase  = phase
        self.run_id = run_id

    def on_epoch_end(self, epoch, logs=None):
        if not self.run_id:
            return
        try:
            import mlflow
            with mlflow.start_run(run_id=self.run_id):
                for k, v in (logs or {}).items():
                    mlflow.log_metric(
                        f"{self.phase}_{k}", float(v), step=epoch + 1
                    )
        except Exception:
            pass


# --------------------------------------------------------------------------- #
class ModelTrainer:
    def __init__(self, num_classes, augmentation_layer,
                 class_names, img_size=IMG_SIZE):
        self.num_classes        = num_classes
        self.augmentation_layer = augmentation_layer
        self.class_names        = class_names
        self.img_size           = img_size
        self.model              = None
        self._wandb_ok          = False
        self._mlflow_run_id     = None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def run(self, train_ds, val_ds, fine_tune=True):
        try:
            log.info("=" * 55)
            log.info("  PHASE 2  -  Model Training  (ResNet101V2)")
            log.info("  Target: accuracy > 60%")
            log.info("=" * 55)

            self.model = self._build()
            self._log_params()

            cw     = self._class_weights(train_ds)
            params = self._hparams(fine_tune)
            log.info(f"Class weights: {cw}")

            self._start_wandb(params)
            self._start_mlflow(params)

            # Phase A: head only
            log.info("-" * 55)
            log.info(f"Phase A: Head training | {EPOCHS_HEAD} epochs max")
            log.info(f"  Trainable params: 558,980")
            log.info("-" * 55)
            hist_a = self._train("HEAD", INITIAL_LR, EPOCHS_HEAD,
                                 train_ds, val_ds, cw)
            best_a = max(hist_a.history["val_accuracy"])
            log.info(f"Phase A complete | best val_acc={best_a:.4f}")

            all_hist = dict(hist_a.history)

            # Phase B: fine-tune
            if fine_tune:
                log.info("-" * 55)
                log.info(f"Phase B: Fine-tune top {FINE_TUNE_LAYERS} backbone layers")
                log.info(f"  Epochs max: {EPOCHS_FINETUNE}  LR: {FINE_TUNE_LR}")
                log.info("-" * 55)
                self._unfreeze()
                hist_b = self._train("FINETUNE", FINE_TUNE_LR,
                                     EPOCHS_FINETUNE, train_ds, val_ds, cw)
                best_b = max(hist_b.history["val_accuracy"])
                log.info(f"Phase B complete | best val_acc={best_b:.4f}")
                for k in hist_b.history:
                    all_hist[k] = all_hist.get(k, []) + hist_b.history[k]

            self._save_history(all_hist)
            self._save_model()
            self._save_tokenizer()

            best = max(all_hist.get("val_accuracy", [0]))
            self._finish_wandb(best)
            self._finish_mlflow(best)

            log.info("=" * 55)
            log.info(f"  Model      -> {MODEL_PATH}")
            log.info(f"  Tokenizer  -> {TOKENIZER_PATH}")
            log.info(f"  History    -> {HISTORY_CSV}")
            log.info(f"  Best val_acc: {best:.4f}  ({best*100:.1f}%)")
            log.info("=" * 55)
            return self.model

        except Exception as e:
            self._end_trackers()
            raise AppException(e, sys) from e

    # ------------------------------------------------------------------ #
    def _build(self) -> tf.keras.Model:
        log.info("Building ResNet101V2 + accuracy-optimised head ...")

        base = tf.keras.applications.ResNet101V2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False
        log.info(f"  {base.name} | layers={len(base.layers)} | frozen=True")

        inputs = tf.keras.Input(shape=(*self.img_size, 3), name="input_image")
        x = self.augmentation_layer(inputs, training=True)
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        x = base(x, training=False)

        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

        x = tf.keras.layers.Dense(
            256, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name="dense_256")(x)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.Dropout(0.40, name="drop1")(x)

        x = tf.keras.layers.Dense(
            128, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name="dense_128")(x)
        x = tf.keras.layers.Dropout(0.30, name="drop2")(x)

        out = tf.keras.layers.Dense(
            self.num_classes, activation="softmax",
            name="predictions")(x)

        model = tf.keras.Model(inputs, out, name="property_classifier")
        log.info("  Head: Dense(256)->BN->Drop(0.4)->Dense(128)->Drop(0.3)->Dense(4)")
        return model

    # ------------------------------------------------------------------ #
    def _train(self, phase, lr, epochs, train_ds, val_ds, cw):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False),
            metrics=["accuracy"],
        )
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(CKPT_DIR / "epoch_{epoch:03d}.keras"),
                save_freq="epoch", verbose=0),
            tf.keras.callbacks.ModelCheckpoint(
                str(CKPT_DIR / "best.keras"),
                monitor="val_accuracy",
                save_best_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=6,
                restore_best_weights=True,
                min_delta=0.005,
                verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=3,
                min_lr=1e-8,
                verbose=1),
            EpochLogger(phase, epochs),
            WandbCallback(phase),
            MLflowCallback(phase, self._mlflow_run_id or ""),
        ]
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=cw,
            verbose=0,
        )

    # ------------------------------------------------------------------ #
    def _unfreeze(self):
        base = next(l for l in self.model.layers
                    if isinstance(l, tf.keras.Model))
        base.trainable = True
        for layer in base.layers[:-FINE_TUNE_LAYERS]:
            layer.trainable = False
        n = sum(1 for l in base.layers if l.trainable)
        t = sum(tf.size(w).numpy() for w in self.model.trainable_weights)
        log.info(f"Unfroze top {n} backbone layers.")
        log.info(f"Phase B trainable params: {t:,}")

    def _class_weights(self, train_ds) -> dict:
        labels = []
        for _, bl in train_ds:
            labels.extend(bl.numpy().tolist())
        labels  = np.array(labels)
        classes = np.unique(labels)
        n       = len(labels)
        return {int(c): n / (len(classes) * np.sum(labels == c))
                for c in classes}

    def _hparams(self, fine_tune) -> dict:
        return {
            "backbone":         "ResNet101V2",
            "head":             "Dense(256)->BN->Drop(0.4)->Dense(128)->Drop(0.3)->Dense(4)",
            "head_trainable":   558980,
            "img_size":         "224x224",
            "batch_size":       32,
            "initial_lr":       INITIAL_LR,
            "fine_tune_lr":     FINE_TUNE_LR,
            "epochs_head":      EPOCHS_HEAD,
            "epochs_finetune":  EPOCHS_FINETUNE if fine_tune else 0,
            "fine_tune_layers": FINE_TUNE_LAYERS,
            "l2_reg":           L2_REG,
            "num_classes":      self.num_classes,
            "classes":          str(self.class_names),
        }

    # ------------------------------------------------------------------ #
    #  WandB key reader                                                    #
    #  Accepts both formats:                                               #
    #    Old: abc123...  (exactly 40 chars)                               #
    #    New: wandb_v1_...  (longer, starts with wandb_v1_)              #
    # ------------------------------------------------------------------ #
    def _get_wandb_key(self) -> str:
        # Source 1: wandb_key.txt
        key_file = Path("wandb_key.txt")
        if key_file.exists():
            raw = key_file.read_text(encoding="utf-8").strip()
            lines = [l.strip() for l in raw.splitlines()
                     if l.strip() and not l.strip().startswith("#")]
            if lines:
                key = lines[0]
                if len(key) >= 40:
                    log.info(f"WandB key loaded from wandb_key.txt "
                             f"(length={len(key)})")
                    return key

        # Source 2: environment variable
        key = os.environ.get("WANDB_API_KEY", "").strip()
        if len(key) >= 40:
            log.info("WandB key loaded from WANDB_API_KEY env var")
            return key

        return ""

    def _start_wandb(self, params: dict):
        try:
            import wandb
            key = self._get_wandb_key()

            if key and len(key) >= 40:
                result = wandb.login(key=key, relogin=True)
                if not result:
                    log.warning("WandB login failed. Check key in wandb_key.txt")
                    return
            else:
                log.warning("WandB key missing or too short.")
                log.warning("  Open wandb_key.txt and paste your key.")
                log.warning("  Get key: https://wandb.ai/authorize")
                log.warning("  Training continues without WandB.")
                return

            wandb.init(
                project=PROJECT_NAME,
                name=f"resnet101v2_{time.strftime('%Y%m%d_%H%M%S')}",
                config=params,
                reinit=True,
            )
            self._wandb_ok = True
            log.info("WandB started -> https://wandb.ai")

        except ImportError:
            log.warning("WandB not installed: python -m pip install wandb")
        except Exception as e:
            log.warning(f"WandB error: {e}")
            log.warning("Training continues without WandB.")

    def _finish_wandb(self, best_acc: float):
        if not self._wandb_ok:
            return
        try:
            import wandb
            if wandb.run:
                wandb.log({"best_val_accuracy": best_acc})
                art = wandb.Artifact("property_classifier", type="model")
                if MODEL_PATH.exists():
                    art.add_file(str(MODEL_PATH))
                if TOKENIZER_PATH.exists():
                    art.add_file(str(TOKENIZER_PATH))
                wandb.log_artifact(art)
                wandb.finish()
                log.info("WandB run finished. Model artifact saved.")
        except Exception as e:
            log.warning(f"WandB finish error: {e}")

    # ------------------------------------------------------------------ #
    #  MLflow - local file tracking, no server or password needed         #
    # ------------------------------------------------------------------ #
    def _start_mlflow(self, params: dict):
        try:
            import pkg_resources
        except ImportError:
            log.warning("MLflow disabled: run  python -m pip install setuptools")
            return

        try:
            import mlflow
            mlruns_dir = Path("mlruns").resolve()
            mlruns_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file:///{str(mlruns_dir)}")
            mlflow.set_experiment(PROJECT_NAME)

            run = mlflow.start_run(
                run_name=f"resnet101v2_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            self._mlflow_run_id = run.info.run_id
            mlflow.log_params(params)
            log.info(f"MLflow tracking locally -> mlruns/")
            log.info(f"  Run ID: {self._mlflow_run_id}")
            log.info("  View: python -m mlflow ui --port 5000")

        except Exception as e:
            log.warning(f"MLflow error: {e}")
            log.warning("Training continues without MLflow.")

    def _finish_mlflow(self, best_acc: float):
        if not self._mlflow_run_id:
            return
        try:
            import mlflow
            with mlflow.start_run(run_id=self._mlflow_run_id):
                mlflow.log_metric("best_val_accuracy", best_acc)
                for path, folder in [
                    (MODEL_PATH,     "model"),
                    (TOKENIZER_PATH, "model"),
                    (HISTORY_CSV,    "logs"),
                ]:
                    if Path(path).exists():
                        mlflow.log_artifact(str(path), folder)
                best_ckpt = CKPT_DIR / "best.keras"
                if best_ckpt.exists():
                    mlflow.log_artifact(str(best_ckpt), "checkpoints")
            mlflow.end_run()
            log.info("MLflow run finished. All artifacts saved.")
        except Exception as e:
            log.warning(f"MLflow finish error: {e}")

    def _end_trackers(self):
        try:
            import wandb
            if wandb.run:
                wandb.finish()
        except Exception:
            pass
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _save_model(self):
        self.model.save(str(MODEL_PATH))
        log.info(f"Model saved -> {MODEL_PATH}")

    def _save_tokenizer(self):
        TOKENIZER_PATH.write_text(json.dumps({
            "class_names":    self.class_names,
            "class_to_index": {c: i for i, c in enumerate(self.class_names)},
            "num_classes":    self.num_classes,
            "input_size":     list(self.img_size),
            "backbone":       "ResNet101V2",
            "head_trainable": 558980,
            "preprocessing":  "resnet_v2_preprocess_input",
        }, indent=2), encoding="utf-8")
        log.info(f"Tokenizer -> {TOKENIZER_PATH}")

    def _save_history(self, history: dict):
        if not history:
            return
        keys = list(history.keys())
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch"] + keys)
            for i, row in enumerate(zip(*[history[k] for k in keys]), 1):
                w.writerow([i] + list(row))
        log.info(f"History -> {HISTORY_CSV}")

    def _log_params(self):
        t = sum(tf.size(w).numpy() for w in self.model.trainable_weights)
        f = sum(tf.size(w).numpy() for w in self.model.non_trainable_weights)
        log.info(f"Parameters:")
        log.info(f"  Trainable    : {t:>12,}")
        log.info(f"  Non-trainable: {f:>12,}")
        log.info(f"  Total        : {t+f:>12,}")
