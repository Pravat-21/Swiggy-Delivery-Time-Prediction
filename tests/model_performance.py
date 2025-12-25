import pandas as pd
import requests
from pathlib import Path
import pytest
import dagshub

import unittest
import mlflow
import os
import pandas as pd
from src.exception import CustomException
from src.logger import file_logging, console_logging
from sklearn.metrics import r2_score, root_mean_squared_error,mean_absolute_error,mean_squared_error
from src.utils import save_data, load_data, load_params, save_model, save_processor,load_model,load_processor
from src.utils import basic_test_data_cleaning
import pickle
#===================================================================================================

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        # dagshub_token = os.getenv("DAGSHUB_PAT")
        # if not dagshub_token:
        #     raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        # os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # dagshub_url = "https://dagshub.com"
        # repo_owner = "Pravat-21"
        # repo_name = "MLops-Mini-Project"

        # # Set up MLflow tracking URI
        # mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        dagshub.init(repo_owner='Pravat-21', repo_name='Swiggy-Delivery-Time-Prediction', mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/Pravat-21/Swiggy-Delivery-Time-Prediction.mlflow")


        # Load the new model from MLflow model registry
        cls.new_model_name = "swiggy_reg"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.sklearn.load_model(cls.new_model_uri)

        # Load the preprocessor
        root = Path(__file__).parent.parent
        processor_path = root/"models"/"processor.pkl"
        cls.preprocessor = load_processor(processor_path)

        # Load holdout test data
        path = root/"data"/"processed"/"processed_test.csv"
        cls.holdout_data = pd.read_csv(path)

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        root = Path(__file__).parent.parent
        test_path = root/"data"/"raw"/"test.csv"
        test_df = pd.read_csv(test_path).dropna()
        
        sample_df = test_df.sample(5,random_state=21)
        clean_df = basic_test_data_cleaning(sample_df)

        input_df = self.preprocessor.transform(clean_df)

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.preprocessor.get_feature_names_out()))

        # Verify the output shape 
        self.assertEqual(len(prediction), input_df.shape[0])

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        mean_error = mean_absolute_error(y_holdout,y_pred_new)
        

        # Define expected thresholds for the performance metrics
        expected_mean_error = 5

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(expected_mean_error, mean_error, f'Accuracy should be at most {expected_mean_error}')
        

if __name__ == "__main__":
    unittest.main()