import os
import pytest
from pathlib import Path
from tensorflow.keras.models import load_model
from cnnClassifier.config.configuration import ConfigurationManager
# Ensure this import matches your exact file structure
from model_architecture.model1_architecture import build_model 
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier import logger

# 1. Test Data Integrity Gate
def test_data_validation_status():
    status_file_path = Path("artifacts/data_validation/status.txt")
    assert status_file_path.exists(), "Data validation status file missing."
    with open(status_file_path, "r") as f:
        status = f.read().split(":")[-1].strip()
    assert status == "True", "Data validation failed!"

# 2. Test Model Compilation and Tensor Shapes for 22 Classes
def test_model_architecture_compilation():
    config_manager = ConfigurationManager()
    transformation_config = config_manager.get_data_transformation_config()
    input_shape = tuple(transformation_config.params_image_size)
    
    # Updated to 22 classes for the skin disease project
    classes = 22 
    
    model = build_model(input_shape=input_shape, classes=classes, learning_rate=0.0001)
    assert model is not None
    assert model.output_shape == (None, classes), f"Expected output shape (None, 22), got {model.output_shape}"

# 3. Inference Pipeline Sanity Check
def test_prediction_pipeline_sanity():
    """
    Verifies that the inference engine can successfully ingest an image,
    preprocess it, and return a valid structural dictionary prediction
    across the 22 skin disease classes.
    """
    # Updated sample image name to match a typical skin lesion sample file
    sample_img_path = "tests/sample_data/lesion_sample.jpg" 
    model_path = "artifacts/training/model.keras"
    
    # Skip the test if structural dependencies are missing locally
    if not os.path.exists(sample_img_path):
        pytest.skip(f"Sample test image not found in {sample_img_path}")
        
    if not os.path.exists(model_path):
        pytest.skip("Model artifact missing from artifacts/training/. Run training first.")
        
    # Load the model artifact once for the test session execution context
    trained_model = load_model(model_path)
        
    # Pass BOTH the file path and the loaded model object into the pipeline
    predictor = PredictionPipeline(filename=sample_img_path, model=trained_model)
    result = predictor.predict()
    
    # Structural assertions matching the new 22-class pipeline schema
    assert isinstance(result, dict), "Prediction output should be a dictionary."
    assert "prediction" in result, "Prediction key missing from output."
    assert "confidence" in result, "Confidence key missing from output."
    assert "class_index" in result, "Class index key missing from output."
    assert isinstance(result["class_index"], int), "Class index must be an integer."
    assert 0 <= result["class_index"] < 22, f"Class index {result['class_index']} is out of bounds for 22 classes."
    
    logger.info("Pytest Gate: Container inference verification successful for skin disease model.")