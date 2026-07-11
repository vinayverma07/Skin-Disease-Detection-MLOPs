import tensorflow as tf
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TrainingConfig
from model_architecture.model1_architecture import build_model
import os
import json

# 3. Model Training Manager
class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self):
        # 1. Load the transformed datasets (Training & Validation)
        train_ds = tf.data.Dataset.load(str(self.config.training_data))
        val_ds = tf.data.Dataset.load(str(self.config.val_data))
        
        # 2. Build the state-of-the-art model
        # Recommended image dimensions for EfficientNetB3 are (300, 300, 3) 
        # Make sure to set IMAGE_SIZE: [300, 300, 3] in your params.yaml if possible
        model = build_model(
            input_shape=tuple(self.config.params_image_size),
            classes=self.config.params_classes,
            learning_rate=0.0001  # Lowered slightly for stable transfer learning
        )
        
        # 3. Fit the model and capture training execution history
        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=self.config.params_epochs
        )

        # Save history dictionary as json file
        history_path = os.path.join(self.config.root_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history.history, f)
        logger.info(f"Training history metrics successfully saved to {history_path}")
        
        # 4. Save model using native Keras v3 format
        model.save(self.config.trained_model_path)
        logger.info(f"Model successfully saved to {self.config.trained_model_path}")