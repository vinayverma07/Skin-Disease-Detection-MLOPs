from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.Data_Validation import DataValidation 
from src.cnnClassifier import logger
from src.cnnClassifier.exception import CustomException
import sys

STAGE_NAME = "Data Ingestion Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):      
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_classes_exist()
        
        

if __name__=='__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj=DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)
    

