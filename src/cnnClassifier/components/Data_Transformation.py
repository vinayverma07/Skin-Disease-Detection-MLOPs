import os
import tensorflow as tf
from pathlib import Path
from src.cnnClassifier import logger
from src.cnnClassifier.config.configuration import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_and_save_data(self):

        logger.info("Loading unified dataset and creating train/validation split...")

        img_size = tuple(self.config.params_image_size[:-1])
        
        # Directly use data_path since it already points to artifacts/data_ingestion/dataset
        data_dir = self.config.data_path

        # Load training split (80%)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=img_size,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            label_mode="categorical"
        )

        # Load validation/test split (20%)
        test_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=img_size,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            label_mode="categorical"
        )

        logger.info(f"Classes found: {train_ds.class_names}")

        # ----------------------------------
        # Data Augmentation (Training Only)
        # ----------------------------------

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1)
        ], name="Augmentation")

        # Normalize pixel values [0,255] -> [-1,1]
        normalization = tf.keras.layers.Rescaling(
            scale=1.0 / 127.5,
            offset=-1
        )

        AUTOTUNE = tf.data.AUTOTUNE

        # Apply augmentation + normalization to training data
        train_ds = train_ds.map(
            lambda x, y: (
                normalization(
                    data_augmentation(x, training=True)
                ),
                y
            ),
            num_parallel_calls=AUTOTUNE
        )

        # Apply normalization only to validation/test data
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