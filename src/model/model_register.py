import json
import mlflow
import sys
import os
import dagshub
from src.exception import CustomException
from pathlib import Path
from src.logger import file_logging, console_logging

#=======================================================================================================
file_logger = file_logging("Model_register_file")
console_logger = console_logging("Model_register_console")
#=======================================================================================================
dagshub.init(repo_owner='Pravat-21', repo_name='Swiggy-Delivery-Time-Prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Pravat-21/Swiggy-Delivery-Time-Prediction.mlflow")
#=======================================================================================================

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)

        file_logger.debug(f'Model info loaded from {file_path}')
        console_logger.debug(f'Model info loaded from {file_path}')

        return model_info
    except FileNotFoundError as f:
        file_logger.error(f'File not found: {file_path} & error is: {f}')
        console_logger.error(f'File not found: {file_path}& error is: {f}')
        raise CustomException(f,sys)
    except Exception as e:
        file_logger.error(f'Unexpected error occurred while loading the model info: {e}')
        console_logger.error(f'Unexpected error occurred while loading the model info: {e}')
        raise CustomException(e,sys)
    
    
def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_name']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        file_logger.info(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
        console_logger.info(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')

    except Exception as e:
        file_logger.error(f'Unexpected error occurred while register_model: {e}')
        console_logger.error(f'Unexpected error occurred while register_model: {e}')
        raise CustomException(e,sys)



def main():
    try:
        root = Path(__file__).parent.parent.parent
        model_info_path = root/"reports"/"models_info.json"

        model_info = load_model_info(model_info_path)
        #model_name = model_info['model_name']
        model_name = "swiggy_reg"

        file_logger.info("Model_info has been fetched")
        console_logger.info("Model_info has been fetched")

        register_model(model_name, model_info)
        

    except Exception as e:
        file_logger.error(f'Unexpected error occurred in main() in register model: {e}')
        console_logger.error(f'Unexpected error occurred in main() in register model: {e}')
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    main()
    print("Model Register has been successfully executed.")
    


