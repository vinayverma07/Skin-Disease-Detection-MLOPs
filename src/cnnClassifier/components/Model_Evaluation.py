import os
import json
import tensorflow as tf
import numpy as np
import mlflow
import mlflow.keras
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.visualization import generate_training_curves, generate_confusion_matrix_plot
import dagshub

load_dotenv()

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate(self):
        """Runs validation calculations, generates assets, and tracking records for 22 skin disease classes."""
        # 1. Load data and model
        logger.info("Loading validation data and trained model assets...")
        test_ds = tf.data.Dataset.load(str(self.config.test_data_path))
        model = tf.keras.models.load_model(str(self.config.model_path))

        # 2. Extract predictions across all classes
        logger.info("Generating predictions across test dataset tensors...")
        y_true = []
        y_pred_probs = []

        for images, labels in test_ds:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            preds = model.predict(images, verbose=0)
            y_pred_probs.extend(preds)

        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred = np.argmax(y_pred_probs, axis=1)

        dagshub.init(repo_owner='vinayverma07', repo_name='Skin-Disease-Detection-MLOPs', mlflow=True)

        # 3. Compute Multi-Class Medical Metrics (Adapted for 22 classes)
        accuracy = np.mean(y_true == y_pred)
        
        # Using 'weighted' average to balance metrics according to the true support of each of the 22 classes
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) # Sensitivity
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Multi-class AUC-ROC calculation using the One-vs-Rest (OVR) strategy
        try:
            auc_roc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate multi-class AUC-ROC (likely due to missing classes in the current batch slice): {e}")
            auc_roc = 0.0

        metrics_dict = {
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall_sensitivity": recall,
            "val_f1_score": f1,
            "val_auc_roc": auc_roc
        }
        
        params_dict = {
            "image_size": str(self.config.params_image_size),  # Convert list to string for MLflow compatibility
            "batch_size": self.config.params_batch_size,
            "num_classes": 22
        }

        # 4. Invoke Isolated Plotting Module
        logger.info("Calling internal plotting utilities to isolate diagnostic assets...")
        generate_training_curves(str(self.config.history_path), str(self.config.plots_dir))
        
        # Dynamically generate 22 class labels (e.g., Class_0 to Class_21)
        # If you have an explicit list of names, replace this with your list of 22 strings
        class_names = [f"Class_{i}" for i in range(22)]
        generate_confusion_matrix_plot(y_true, y_pred, classes=class_names, save_dir=str(self.config.plots_dir))

        # 5. Connect to MLflow remote server via DagsHub
        if self.config.mlflow_uri != "":
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            
        # Ensure experiment exists for skin disease tracking
        mlflow.set_experiment("Skin_Disease_Classification_22_Classes")

        with mlflow.start_run():
            logger.info("Logging runtime parameters and diagnostic metrics to MLflow...")
            
            # Log Parameters
            mlflow.log_params(params_dict)
            
            # Log Metrics
            mlflow.log_metrics(metrics_dict)
            
            # Log Plots as Artifacts
            mlflow.log_artifacts(str(self.config.plots_dir), artifact_path="evaluation_plots")
            
            # Log Model Architecture
            mlflow.keras.log_model(model, "model", registered_model_name="Skin_Disease_EfficientNet_Model")
            
        logger.info("Evaluation metrics successfully tracked and artifacts archived.")