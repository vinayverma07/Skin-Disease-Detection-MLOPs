import os
import tensorflow as tf
from pathlib import Path
from src.cnnClassifier import logger
from src.cnnClassifier.config.configuration import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_and_save_data(self):

        logger.info("Loading train and test datasets...")

        img_size = tuple(self.config.params_image_size[:-1])

        train_path = os.path.join(self.config.data_path, "train")
        test_path = os.path.join(self.config.data_path, "test")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            image_size=img_size,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            label_mode="categorical"
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_path,
            image_size=img_size,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            label_mode="categorical"
        )

        logger.info(f"Classes: {train_ds.class_names}")

        # ----------------------------------
        # Data Augmentation (Training Only)
        # ----------------------------------

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03)      # ≈10°
        ])

        # Normalize [0,255] -> [-1,1]
        normalization = tf.keras.layers.Rescaling(
            scale=1.0 / 127.5,
            offset=-1
        )

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.map(
            lambda x, y: (
                normalization(
                    data_augmentation(x, training=True)
                ),
                y
            ),
            num_parallel_calls=AUTOTUNE
        )

        test_ds = test_ds.map(
            lambda x, y: (
                normalization(x),
                y
            ),
            num_parallel_calls=AUTOTUNE
        )

        train_ds = train_ds.prefetch(AUTOTUNE)
        test_ds = test_ds.prefetch(AUTOTUNE)

        logger.info("Saving transformed training dataset...")
        tf.data.Dataset.save(
            train_ds,
            str(self.config.train_dataset_path)
        )

        logger.info("Saving transformed testing dataset...")
        tf.data.Dataset.save(
            test_ds,
            str(self.config.val_dataset_path)
        )

        logger.info("Data transformation completed successfully.")