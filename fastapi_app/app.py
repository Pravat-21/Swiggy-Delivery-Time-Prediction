from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
import uvicorn
import pandas as pd
import mlflow
from fastapi.responses import StreamingResponse
import json
import io
import pickle
from mlflow import MlflowClient
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn import set_config
import dagshub
from src.exception import CustomException
import sys
from pathlib import Path
from src.logger import file_logging, console_logging
from src.utils import load_processor,basic_test_data_cleaning
#=============================================================================
#=======================================================================================================
file_logger = file_logging("Model_register_file")
console_logger = console_logging("Model_register_console")
#=======================================================================================================
#load_dotenv()
dagshub.init(repo_owner='Pravat-21', repo_name='Swiggy-Delivery-Time-Prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Pravat-21/Swiggy-Delivery-Time-Prediction.mlflow")

#=======================================================================================================

class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

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
    
def get_latest_model_version(model_name):
    """This function helps us to get the latest version of model from model registry"""

    try:
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["Staging"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        
        file_logger.info("Latest version of model has been fetched")
        console_logger.info("Latest version of model has been fetched")

        return latest_version[0].version if latest_version else None
    
    except Exception as e:
        file_logger.error(f'Unexpected error occurred while get_latest_model_version: {e}')
        console_logger.error(f'Unexpected error occurred while get_latest_model_version: {e}')
        raise CustomException(e,sys)
    
root = Path(__file__).parent.parent

processor_path = root/"models"/"processor.pkl"

processor = load_processor(processor_path)

model_name = "swiggy_reg"
model_version = get_latest_model_version(model_name)
model_stage = "Staging"
print(model_version)
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)
#model = mlflow.pyfunc.load_model(model_uri)

model_pipe = Pipeline(steps=[
('preprocess',processor),
("regressor",model)
])
    
app = FastAPI()

@app.get("/")
def home():
    return "Hello, welcome to swiggy time prediciton page"

@app.post(path="/predict")
def do_predictions(data: Data):
    pred_data = pd.DataFrame({
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
        },index=[0]
    )

    cleaned_data = basic_test_data_cleaning(pred_data)
    df =cleaned_data.dropna()

    if len(df) != 0:

        predictions = model_pipe.predict(df)[0]
        return predictions
    else:
        return "Please, check all the columns. You might miss to put some values."



@app.post("/bulk-prediciton")
async def predict_bulk_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    clean_df = basic_test_data_cleaning(df)
    clean_df = clean_df.dropna()

    clean_df["prediction"] = model_pipe.predict(clean_df)

    stream = io.StringIO()
    clean_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )


if __name__ == "__main__":
    uvicorn.run(app="app:app",host='0.0.0.0',port=8000)



    



