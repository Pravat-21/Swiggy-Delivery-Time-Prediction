import pandas as pd
import numpy as np
import os
import sys
import yaml
import pickle
from src.logger import file_logging, console_logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator as Model
import haversine as hs

file_logger = file_logging("src_Utils_file")
con_logger = console_logging("src_Utils_console")

def time_diff(row):
    try:
        #logic of function
        order_picked = pd.to_timedelta(row['Time_Order_picked'])
        time_order =  pd.to_timedelta(row['Time_Orderd'])
        x = order_picked - time_order
        return x
    
    except Exception as e:
        file_logger.error(f"Error has been occured into time_diff function in utils & the error is {e}")
        con_logger.error(f"Error has been occured into time_diff function in utils& the error is {e}")
        raise CustomException(e,sys)

def haversine_row(row):
    try:
        # logic of function
        loc1 = (row['Restaurant_latitude'], row['Restaurant_longitude'])
        loc2 = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
        hvr_distance = hs.haversine(loc1, loc2, unit=hs.Unit.KILOMETERS)
        return hvr_distance
    
    except Exception as e:
        file_logger.error(f"Error has been occured into haversine_row function in utils & the error is {e}")
        con_logger.error(f"Error has been occured into haversine_row function in utils & the error is {e}")
        raise CustomException(e,sys)


def basic_test_data_cleaning(df):
    try:
        file_logger.info("Creating basic_test_data_cleaning function in utils.....")
        con_logger.info("Creating basic_test_data_cleaning function in utils.....")

        # logic of the function
        df.drop(columns=['ID'],inplace=True)
        df['Restaurant_latitude'] = abs(df['Restaurant_latitude'].replace(0,np.nan))
        df['Restaurant_longitude'] = abs(df['Restaurant_longitude'].replace(0,np.nan))
        cols = ["Delivery_location_latitude","Delivery_location_longitude"]
        df.loc[df['Restaurant_latitude'].isnull(), cols] = np.nan

        columns = ["Delivery_person_Age",  "Delivery_person_Ratings","Time_Orderd", "Weatherconditions", "Road_traffic_density",
        "multiple_deliveries", "Festival", "City"]
        for i in columns:
            df[i] = df[i].replace('NaN ',np.nan)
        df['Weatherconditions'] = df['Weatherconditions'].replace("conditions NaN",np.nan)
        df['Weatherconditions'] = df['Weatherconditions'].apply(lambda x: x.split()[-1] if isinstance(x, str) else x)
        #df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: x[-2:]).astype("float")

        # changing datatype 
        df['Delivery_person_Age'] = df['Delivery_person_Age'].astype("float")
        df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype("float")
        df['Order_Date'] = pd.to_datetime(df['Order_Date'],dayfirst=True)
        df['multiple_deliveries'] = df['multiple_deliveries'].astype("float")

        # applying harversine distance
        df['Distance_res_to_loc_KM'] = df[['Restaurant_latitude', 'Restaurant_longitude','Delivery_location_latitude', 'Delivery_location_longitude']].apply(haversine_row,axis=1)
        df.insert(7,"Distance_res_loc_KM",df.pop('Distance_res_to_loc_KM'))

        # finding the time taken between restrurant & order pickup
        df["Time_res_pickup"] = (df[['Time_Order_picked','Time_Orderd']].apply(time_diff,axis=1).dt.total_seconds()/60).replace({-1425.:15., -1430.:10., -1435.:5.})
        df.insert(11,"Time_res_pickup",df.pop("Time_res_pickup"))
        df["order_time_hr"] = pd.to_datetime(df['Time_Orderd']).dt.hour
        df.insert(10,"order_time_hr",df.pop("order_time_hr"))
        df["time_of_day"] = pd.cut(
        df['order_time_hr'],
        bins=[0, 6, 12, 17, 20, 24],
        labels=['after midnight', 'morning', 'afternoon', 'evening', 'night'],
        right=False
        )
        df.insert(11,"time_of_day",df.pop("time_of_day"))

        # fixing values in columns:
        df['Road_traffic_density'] = df['Road_traffic_density'].str.strip().str.lower()
        df['Weatherconditions'] = df['Weatherconditions'].str.strip().str.lower()
        df['Type_of_order'] = df['Type_of_order'].str.strip().str.lower()
        df['Type_of_vehicle'] = df['Type_of_vehicle'].str.strip().str.lower()
        df['Festival'] = df['Festival'].str.strip().str.lower()
        df['City'] = df['City'].str.strip().str.lower()

        # dropping some rows
        df.drop(index=df[df['Delivery_person_Age'] <18].index,inplace=True)
        df.drop(index=df[df['Delivery_person_Ratings']>5].index,inplace=True)
        
        # creating rider city info
        rename_city = {"INDO":"Indore","BANG":"Bengaluru","COIMB":"Coimbatore","CHEN":"Chennai","HYD":"Hyderabad",
                "RANCHI":"Ranchi","MYS":"Mysore","DEH":"Dehradun","KOC":"Kochi","PUNE":"Pune","LUD":"Ludhiana",
                "LUDH":"Ludhiana","KNP":"Kanpur","MUM":"Mumbai","KOL":"Kolkata","JAP":"Jamshedpur","SUR":"Surat",
                "GOA":"Goa","AURG":"Aurangabad","AGR":"Agra","VAD":"Vadodara","ALH":"Allahabad","BHP":"Bhopal"}
        
        df['rider_city'] = df['Delivery_person_ID'].str.split("RES").str.get(0).replace(rename_city)
        df.insert(1,'rider_city',df.pop('rider_city'))
        df.insert(11,'order_dayname',df['Order_Date'].dt.day_name())
        df.insert(12,'order_in_weekend',np.where(df['order_dayname'].isin(['Saturday','Sunday']),1,0))
        

        df.drop(columns=['Delivery_person_ID'],inplace=True)

        # after doing EDA these columns are selected to be removed
        df.drop(columns=['rider_city','Type_of_order','order_in_weekend','order_time_hr','Time_res_pickup','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Order_Date','Time_Orderd','Time_Order_picked'],inplace=True)

        df = df.dropna()

        file_logger.info("Successfully created basic_test_data_cleaning function.")
        con_logger.info("Successfully created basic_test_data_cleaning function.")
        
        return df
    
    except Exception as e:
        file_logger.error(f"Error has been occured into basic_test_data_cleaning function in utils & the error is {e}")
        con_logger.error(f"Error has been occured into basic_test_data_cleaning function in utils & the error is {e}")
        raise CustomException(e,sys)


def save_data(df:pd.DataFrame, path:str)-> None:
    """This function helps to save csv data in the given folder with given file name"""

    file_logger.info("Now in save_data function from utils.py")
    con_logger.info("Now in save_data function from utils.py")

    try:
        os.makedirs(os.path.dirname(path),exist_ok=True)
        df.to_csv(path,index=False)

        file_logger.info(f"successfully save the data into {path} as csv")
        con_logger.info(f"successfully save the data into {path} as csv")

    except Exception as e:
        file_logger.error("Error has been occured in save_data function from utils.py")
        con_logger.error("Error has been occured in save_data function from utils.py")
        raise CustomException(e,sys)
    
def load_data(path:str)->pd.DataFrame:
    """Using this function we can load data located into given data path."""

    try:
        file_logger.info("Now in load_data function from utils.py")
        con_logger.info("Now in load_data function from utils.py")

        df = pd.read_csv(path)

        file_logger.info(f"successfully load the data from the {path}")
        con_logger.info(f"successfully load the data from the {path}")

        return df
    
    except Exception as e:
        file_logger.error("Error has been occured in load_data function from utils.py")
        con_logger.error("Error has been occured in load_data function from utils.py")
        raise CustomException(e,sys)
    
def load_params()->dict:
    """Using this function we can load our params.yaml file for parameter usages"""

    file_logger.info("Now in load_params function from utils.py")
    con_logger.info("Now in load_params function from utils.py")

    try:
        with open("params.yaml","rb") as file:
            params=yaml.safe_load(file)

            file_logger.info("successfully load all the parameters.")
            con_logger.info("successfully load all the parameters.")

            return params
    except Exception as e:
        file_logger.error("Error has been occured in load_params function from utils.py")
        con_logger.error("Error has been occured in load_params function from utils.py")
        raise CustomException(e,sys)
    
def save_processor(processor:Pipeline, path:str)->None:
    """This function saves processor in given file path"""

    file_logger.info("Now in save_processor function from utils.py")
    con_logger.info("Now in save_processor function from utils.py")

    try:
        with open(path,'wb') as file:
            pickle.dump(processor,file)

        file_logger.info("successfully dumped the processor.")
        con_logger.info("successfully dumped the processor.")

    except Exception as e:
        file_logger.error("Error has been occured in save_processor function from utils.py")
        con_logger.error("Error has been occured in save_processor function from utils.py")
        raise CustomException(e,sys)
    
def save_model(model:Model, path:str)->None:
    """This function saves model in given file path"""

    file_logger.info("Now in save_model function from utils.py")
    con_logger.info("Now in save_model function from utils.py")

    try:
        with open(path,'wb') as file:
            pickle.dump(model,file)

        file_logger.info("successfully dumped the model.")
        con_logger.info("successfully dumped the model.")

    except Exception as e:
        file_logger.error("Error has been occured in save_model function from utils.py")
        con_logger.error("Error has been occured in save_model function from utils.py")
        raise CustomException(e,sys)

def load_model(file_path: str)->Model:
    """Load the trained model from a file."""
    
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)

        file_logger.info(f'Model loaded from {file_path}')
        con_logger.info(f'Model loaded from {file_path}')
        return model
    except Exception as e:
        file_logger.error(f"In load_model() from utils.py, error has been ocurred & error is {e}")
        con_logger.error(f"In  load_model() from utils.py, error has been ocurred & error is {e}")
        raise CustomException(e,sys)

def load_processor(file_path: str)->Pipeline:
    """Load the processor for transformation from a file."""

    try:
        with open(file_path, 'rb') as file:
            processor = pickle.load(file)

        file_logger.info(f'processor loaded from {file_path}')
        con_logger.info(f'processor loaded from {file_path}')
        return processor
    except Exception as e:
        file_logger.error(f"In utils.py, in load_processor() error has been ocurred & error is {e}")
        con_logger.error(f"In utils.py, in load_processor() error has been ocurred & error is {e}")
        raise CustomException(e,sys)

