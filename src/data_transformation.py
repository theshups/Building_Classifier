"""src/data_transformation.py - tf.data pipeline"""
import sys
from pathlib import Path
import tensorflow as tf
import tensorflow.data  # force submodule load on Windows

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger import get_logger
from exception import AppException

log = get_logger(__name__)
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
AUTOTUNE   = -1  # equals tf.data.AUTOTUNE


class DataTransformation:
    def __init__(self, train_dir, val_dir, test_dir,
                 img_size=(224,224), batch_size=32):
        self.train_dir  = Path(train_dir)
        self.val_dir    = Path(val_dir)
        self.test_dir   = Path(test_dir)
        self.img_size   = img_size
        self.batch_size = batch_size

    def run(self):
        try:
            log.info("=" * 55)
            log.info("  PHASE 1  -  Data Transformation")
            log.info("=" * 55)
            train_raw, class_names = self._load(self.train_dir, shuffle=True)
            val_raw,   _           = self._load(self.val_dir,   shuffle=False)
            test_raw,  _           = self._load(self.test_dir,  shuffle=False)
            log.info(f"Classes ({len(class_names)}): {class_names}")
            train_ds = self._pipeline(train_raw)
            val_ds   = self._pipeline(val_raw)
            test_ds  = self._pipeline(test_raw)
            log.info("tf.data pipelines ready [cache + prefetch]")
            return train_ds, val_ds, test_ds, class_names
        except Exception as e:
            raise AppException(e, sys) from e

    def get_augmentation_layer(self) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.20),
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
        ], name="augmentation")

    def _load(self, directory: Path, shuffle: bool):
        if not directory.exists():
            raise AppException(f"Directory not found: {directory}", sys)
        ds = tf.keras.utils.image_dataset_from_directory(
            str(directory), image_size=self.img_size,
            batch_size=self.batch_size, shuffle=shuffle,
            seed=42, label_mode="int",
        )
        return ds, ds.class_names

    def _pipeline(self, ds):
        return ds.map(self._normalize, num_parallel_calls=AUTOTUNE)\
                 .cache().prefetch(AUTOTUNE)

    @staticmethod
    def _normalize(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels
