import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from src.cnnClassifier import logger

class PredictionPipeline:
    def __init__(self, filename, model, target_size=(224, 224)):
        """
        Initializes the pipeline for 22-class skin disease inference.
        Note: EfficientNetB3 performs best at (300, 300). Adjust target_size 
        to match your exact configuration if different.
        """
        self.filename = filename
        self.model = model  # Pass the pre-loaded global model here
        self.target_size = target_size
        
        # Define the exact names of your 22 classes.
        # CRITICAL: Order these exactly matching how your Data Generator sorted the folders.
        self.class_mapping = {
            0: "Actinic Keratosis",
            1: "Basal Cell Carcinoma",
            2: "Dermatofibroma",
            3: "Melanoma",
            4: "Nevus",
            5: "Squamous Cell Carcinoma",
            6: "Vascular Lesion",
            7: "Atopic Dermatitis",
            8: "Eczema",
            9: "Psoriasis",
            10: "Seborrheic Keratosis",
            11: "Tinea Ringworm",
            12: "Warts Molluscum",
            13: "Urticaria Hives",
            14: "Impetigo",
            15: "Acne Vulgaris",
            16: "Rosacea",
            17: "Lichen Planus",
            18: "Lupus Erythematosus",
            19: "Vitiligo",
            20: "Shingles Herpes Zoster",
            21: "Chickenpox"
        }

    def predict(self):
        logger.info("Executing skin disease classification from model cache...")
        
        # 1. Preprocess image
        # EfficientNet models handle rescaling internally inside Keras, 
        # but scaling to [0, 1] depends on how your Data Ingestion / Transformation pipeline formatted tensors.
        test_image = load_img(self.filename, target_size=self.target_size)
        test_image = img_to_array(test_image)
        
        # Check if you manually rescaled by 1/255.0 during training. If so, leave this enabled:
        test_image = test_image / 255.0 
        test_image = np.expand_dims(test_image, axis=0)

        # 2. Run inference using the cached model
        predictions = self.model.predict(test_image, verbose=0)
        
        # Find index with highest probability score
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class_idx])

        # 3. Resolve the label name securely from dictionary
        prediction_label = self.class_mapping.get(
            predicted_class_idx, 
            f"Unknown Skin Condition (Index {predicted_class_idx})"
        )

        logger.info(f"Prediction calculated: {prediction_label} ({confidence:.2%})")
        
        return {
            "prediction": prediction_label,
            "confidence": f"{confidence:.2%}",
            "class_index": int(predicted_class_idx)
        }