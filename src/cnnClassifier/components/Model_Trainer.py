import tensorflow as tf
from pathlib import Path
from src.cnnClassifier import logger
from src.cnnClassifier.entity.config_entity import TrainingConfig
from model_architecture.model1_architecture import build_model
import os
import json

# 3. Model Training Manager
class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self):
        """
        Loads transformed datasets and trains the model.
        """
        logger.info("Loading transformed training and validation datasets...")

        # Load saved tf.data.Dataset objects from data transformation
        train_ds = tf.data.Dataset.load(str(self.config.training_data))
        val_ds = tf.data.Dataset.load(str(self.config.val_data))

        # Dynamically calculate class count from one-hot target shape
        for _, labels in train_ds.take(1):
            num_classes = labels.shape[-1]
            break

        logger.info(f"Detected {num_classes} output classes.")

        model = build_model(self,num_classes=num_classes)

        logger.info("Starting model training...")

        # Callbacks for best model saving and early stopping
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.trained_model_path),
                save_best_only=True,
                monitor="val_loss",
                mode="min"
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        ]

        # Model fit
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.params_epochs,
            callbacks=callbacks
        )

        logger.info(f"Saving final trained model to: {self.config.trained_model_path}")
        model.save(str(self.config.trained_model_path))

        logger.info("Model training stage completed successfully.")