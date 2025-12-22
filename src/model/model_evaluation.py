import pandas as pd
import yaml
import pickle
import os
import json
import sys
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.base import BaseEstimator as Model
from pathlib import Path
from src.exception import CustomException
from src.logger import file_logging, console_logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, root_mean_squared_error,mean_absolute_error,mean_squared_error
from src.utils import save_data, load_data, load_params, save_model, save_processor,load_model
#=======================================================================================================
file_logger = file_logging("Model_evaluation_file")
console_logger = console_logging("Model_evaluation_console")
#=======================================================================================================
dagshub.init(repo_owner='Pravat-21', repo_name='Swiggy-Delivery-Time-Prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Pravat-21/Swiggy-Delivery-Time-Prediction.mlflow")

mlflow.set_experiment("DVC-Pipeline-Model Evaluation")
#=======================================================================================================

def adjusted_r2_score(r2, n_samples, n_features):
    """
    Calculate Adjusted R² score

    Parameters
    ----------
    r2 : float
        R² score
    n_samples : int
        Number of observations
    n_features : int
        Number of predictors

    Returns
    -------
    float
        Adjusted R²
    """
    try:
        adj_r2 =  1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
        return adj_r2
    
    except Exception as e:
        file_logger.error(f"Error has been occured into adjusted_r2_score() function in model evaluation module & the error is {e}")
        console_logger.error(f"Error has been occured into adjusted_r2_score() function in model evaluation module & the error is {e}")
        raise CustomException(e,sys)


def evaluation_model(model:Model, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.Series,y_test:pd.Series):
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_mae = mean_absolute_error(y_train,y_train_pred)
        test_mae = mean_absolute_error(y_test,y_test_pred)

        train_r2 = r2_score(y_train,y_train_pred)
        test_r2 = r2_score(y_test,y_test_pred)

        train_adj_r2 = adjusted_r2_score(train_r2, X_train.shape[0],X_train.shape[1])
        test_adj_r2 = adjusted_r2_score(test_r2, X_test.shape[0],X_test.shape[1])

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        mean_cv_score = -(cv_scores.mean())

        params_dict = {"train_mae":train_mae, "test_mae":test_mae, 
                    "train_r2":train_r2, "test_r2":test_r2,
                    "train_adj_r2":train_adj_r2, "test_adj_r2":test_adj_r2, 
                    "mean_cv_score":mean_cv_score}

        return params_dict, list(cv_scores)
    
    except Exception as e:
        file_logger.error(f"Error has been occured into evaluation_model() function in model evaluation module & the error is {e}")
        console_logger.error(f"Error has been occured into evaluation_model() function in model evaluation module & the error is {e}")
        raise CustomException(e,sys)
    
def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""

    file_logger.info("In save_metrics function from model_evaluation module....")
    console_logger.info("In save_metrics function from model_evaluation module....")

    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)

        file_logger.info(f'Metrics saved to {file_path}')
        console_logger.info(f'Metrics saved to {file_path}')

    except Exception as e:
        file_logger.error(f"In save_metrics function from model Evaluation, error has been ocurred & error is {e}")
        console_logger.error(f"In save_metrics function from model Evaluation, error has been ocurred & error is {e}")
        raise CustomException(e,sys)
    
def save_model_info(run_id: str, model_name: str, file_path: str, artifact_path:str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_name': model_name,"artifact_path":artifact_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        file_logger.info(f'Model info saved to {file_path}')
                         
    except Exception as e:
        file_logger.error(f"In save_model_info function from model Evaluation, error has been ocurred & error is {e}")
        console_logger.error(f"In save_model_info function from model Evaluation, error has been ocurred & error is {e}")
        raise CustomException(e,sys)
    
#------------------------------------------------------------------------

def main():

    with mlflow.start_run(run_name="Model Evaluation in DVC pipeline") as run:

        try:
            mlflow.set_tags(
            {
                "Logging-type":"Information logging for model evaluation",
                "Model":"Swiggy Delivery Time Regressor",
                "author-name":"Pravat",
                "Description":"Here Detailed information of Final_estimator(stacking) has been logged."
            }
            )

            root = Path(__file__).parent.parent.parent
            train_data_path = root/"data"/"processed"/"processed_train.csv"
            test_data_path = root/"data"/"processed"/"processed_test.csv"
            model_path = root/"models"/"final_model.pkl"

            train_df = load_data(train_data_path)
            test_df = load_data(test_data_path)

            X_train = train_df.iloc[:,:-1]
            y_train = train_df.iloc[:,-1]

            X_test = test_df.iloc[:,:-1]
            y_test = test_df.iloc[:,-1]

            model = load_model(model_path)
            mlflow.log_params(model.get_params())

            eval_matrics,cv_scores = evaluation_model(model, X_train,X_test,y_train,y_test)

            matrics_path = root/"reports"/"eval_metrics.json"
            os.makedirs(os.path.dirname(matrics_path),exist_ok=True)
            save_metrics(eval_matrics, matrics_path)

            mlflow.log_metrics(eval_matrics)
            mlflow.log_metrics({f"CV-{num}": -score for num, score in enumerate(cv_scores)})

            model_signature = mlflow.models.infer_signature(model_input=X_train.sample(10,random_state=42),
                                         model_output=model.predict(X_test.sample(10,random_state=42)))
            
            mlflow.sklearn.log_model(model,"Swiggy_delivery_time_prediction_model",signature=model_signature)


            stacking_path = root/"models"/"stacking_regressor.pkl"
            processor_path = root/"models"/"processor.pkl"
            transformer_path = root/"models"/"stacking_transformer.pkl"

            mlflow.log_artifact(stacking_path)
            mlflow.log_artifact(processor_path)
            mlflow.log_artifact(transformer_path)
            mlflow.log_artifact(matrics_path)

            model_info_path = root/"reports"/"models_info.json"
            os.makedirs(os.path.dirname(model_info_path),exist_ok=True)

            model_name = "Swiggy_delivery_time_prediction_model"
            artifacts_path = mlflow.get_artifact_uri()

            save_model_info(run_id=run.info.run_id, model_name= model_name, file_path = model_info_path,artifact_path=artifacts_path)

        except Exception as e:
            file_logger.error(f"In save_model_info function from model Evaluation, error has been ocurred & error is {e}")
            console_logger.error(f"In save_model_info function from model Evaluation, error has been ocurred & error is {e}")
            raise CustomException(e,sys)


if __name__ == "__main__":
    main()
    print("Model Evaluation has been successfully executed.")














        






