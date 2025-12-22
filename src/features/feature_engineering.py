import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, root_mean_squared_error,mean_absolute_error,mean_squared_error
from src.exception import CustomException
from src.logger import file_logging, console_logging
from src.utils import load_data, save_data, load_params, save_processor
from sklearn import set_config
#======================================================================================

file_logger = file_logging("Feature_Engineering_file")
console_logger = console_logging("Feature_Engineering_console")

# set the transformer outputs to pandas
set_config(transform_output='pandas')

#======================================================================================

def create_processor()-> Pipeline:
    """Here this function creates the required processor for tranform data."""

    file_logger.info("Now in create_processor() function in feature engineering.py")
    console_logger.info("Now in create_processor() function in feature engineering.py")

    try:
        nom_cat_cols = ['order_dayname','time_of_day','Type_of_vehicle','Weatherconditions','Festival','City']
        ord_cat_cols = ['Road_traffic_density']
        trafic_list = ['low','medium','high','jam']

        trasformer = ColumnTransformer(
                transformers=[
                    ("OHE",OneHotEncoder(drop='first',sparse_output=False),nom_cat_cols),
                    ("Ordinal",OrdinalEncoder(categories=[trafic_list],handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-2),ord_cat_cols)
                ],remainder="passthrough"
            )
        
        processor = Pipeline(
        steps=[
            ("First_transformation",trasformer),
            ("scaling",StandardScaler())
        ]
        )
        file_logger.info("Successfully created processor")
        console_logger.info("Successfully created processor")

        return processor
    
    except Exception as e:
        file_logger.error(f"Error has been occured into create_processor function in feature_engineering module & the error is {e}")
        console_logger.error(f"Error has been occured into create_processor function in feature_engineering module & the error is {e}")
        raise CustomException(e,sys)
    
def drop_missing_vals(df:pd.DataFrame)->pd.DataFrame:
    """This function removes all the NaN values from dataset"""

    file_logger.info("Now in drop_missing_vals() function in feature engineering.py")
    console_logger.info("Now in drop_missing_vals() function in feature engineering.py")

    try:
        dropna_df = df.dropna()

        file_logger.info(f"Successfully dropped all the missing values & now the shape of the data is {dropna_df.shape}")
        console_logger.info(f"Successfully dropped all the missing values & now the shape of the data is {dropna_df.shape}")

        return dropna_df
    
    except Exception as e:
        file_logger.error(f"Error has been occured into drop_missing_vals function in feature_engineering module & the error is {e}")
        console_logger.error(f"Error has been occured into drop_missing_vals function in feature_engineering module & the error is {e}")
        raise CustomException(e,sys)
    
def data_loading_splitting():
    """This function helps to load data & drop all the missing values then splits into X_train,X_test,y_train,y_test then return these four dataframe,series"""

    file_logger.info("Now in data_loading_splitting() function in feature engineering.py")
    console_logger.info("Now in data_loading_splitting() function in feature engineering.py")

    try:
        root = Path(__file__).parent.parent.parent
        train_data_path = root/"data"/"cleaned"/"clean_train.csv"
        test_data_path = root/"data"/"cleaned"/"clean_test.csv"

        train_df = load_data(train_data_path)

        file_logger.info(f"train data is loaded & the shape is{train_df.shape}")
        console_logger.info(f"train data is loaded & the shape is{train_df.shape}")

        train_df = drop_missing_vals(train_df)

        file_logger.info(f"Missing values are removed & the now train data-shape is{train_df.shape}")
        console_logger.info(f"Missing values are removed & the now train data-shape is{train_df.shape}")

        test_df = load_data(test_data_path)

        file_logger.info(f"test data is loaded & the shape is{test_df.shape}")
        console_logger.info(f"test data is loaded & the shape is{test_df.shape}")

        test_df = drop_missing_vals(test_df)

        file_logger.info(f"Missing values are removed & the now test data-shape is{test_df.shape}")
        console_logger.info(f"Missing values are removed & the now test data-shape is{test_df.shape}")

        X_train = train_df.iloc[:,:-1]
        y_train = train_df.iloc[:,-1]

        X_test = test_df.iloc[:,:-1]
        y_test = test_df.iloc[:,-1]

        file_logger.info(f"Successfully split the dataset & shape is:\nX_train->{X_train.shape}\nX_test->{X_test.shape}\ny_train->{y_train.shape}\ny_test->{y_test.shape}")
        console_logger.info(f"Successfully split the dataset & shape is:\nX_train->{X_train.shape}\nX_test->{X_test.shape}\ny_train->{y_train.shape}\ny_test->{y_test.shape}")

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        file_logger.error(f"Error has been occured into data_loading_splitting function in feature_engineering module & the error is {e}")
        console_logger.error(f"Error has been occured into data_loading_splitting function in feature_engineering module & the error is {e}")
        raise CustomException(e,sys)


def join_X_and_y(X: pd.DataFrame, y: pd.Series):
    """This function joins one series with a dataframe"""
    # join based on indexes
    joined_df = X.join(y,how='inner')
    return joined_df


def main():

    try:
        X_train,X_test,y_train,y_test = data_loading_splitting()
        processor = create_processor()
        X_train_trns = processor.fit_transform(X_train)
        X_test_trns = processor.transform(X_test)

        train_transform_df = join_X_and_y(X_train_trns,y_train)
        test_transform_df = join_X_and_y(X_test_trns,y_test)

        file_logger.info(f"train_transform data looks like{train_transform_df.head(1)}")
        file_logger.info(f"test_transform data looks like{test_transform_df.head(1)}")
        console_logger.info(f"train_transform data looks like{train_transform_df.head(1)}")
        console_logger.info(f"test_transform data looks like{test_transform_df.head(1)}")

        root = Path(__file__).parent.parent.parent
        train_path = root/"data"/"processed"/"processed_train.csv"
        test_path = root/"data"/"processed"/"processed_test.csv"
        processor_path = root/"models"/"processor.pkl"

        save_data(train_transform_df,train_path)
        save_data(test_transform_df,test_path)
        save_processor(processor,processor_path)
    
    except Exception as e:
        file_logger.error(f"Error has been occured into main() function in feature_engineering module & the error is {e}")
        console_logger.error(f"Error has been occured into main() function in feature_engineering module & the error is {e}")
        raise CustomException(e,sys)
    
#==========================================================================
if __name__ == "__main__":
    main()
    print("Feature engineering module is sucessfully executed")




    

