import pandas as pd
import numpy as np
import os
import sys
import yaml
from src.logger import file_logging, console_logging
from src.exception import CustomException

file_logger = file_logging("src_Utils_file")
con_logger = console_logging("src_Utils_console")


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

