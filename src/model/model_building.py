import pandas as pd
import yaml
import pickle
import os
import sys

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path
from sklearn.ensemble import StackingRegressor
from sklearn.base import BaseEstimator as Model

from src.exception import CustomException
from src.logger import file_logging, console_logging
from src.utils import save_data, load_data, load_params, save_model, save_processor

#==========================================================================================
file_logger = file_logging("Model_building_file")
console_logger = console_logging("Model_building_console")
#==========================================================================================

def model_creation()->Model:
    """This function helps to build our regressor model with power transformation using stacking concept."""

    try:
        params = load_params()['train']

        rf_params = params['Random_Forest']
        file_logger.info("Random Forest parameters are loaded")
        console_logger.info("Random Forest parameters are loaded")

        lgbm_params = params['LGBM']
        file_logger.info("LGBM parameters are loaded")
        console_logger.info("LGBM parameters are loaded")

        best_RF = RandomForestRegressor(**rf_params)
        file_logger.info("RF Model has been created")
        console_logger.info("RF Model has been created")

        best_LGBM = LGBMRegressor(**lgbm_params)
        file_logger.info("LGBM Model has been created")
        console_logger.info("LGBM Model has been created")

        st_reg = StackingRegressor(estimators=[
        ("RF",best_RF),
        ("LGBM",best_LGBM)
        ],
        final_estimator=LinearRegression(),cv=5,n_jobs=-1)
        file_logger.info("Stacking regressor has been created")
        console_logger.info("Stacking regressor has been created")

        model = TransformedTargetRegressor(regressor=st_reg, transformer=PowerTransformer())
        file_logger.info("Final regressor with power transformation has been created")
        console_logger.info("Final regressor with power transformation has been created")

        return model
    
    except Exception as e:
        file_logger.error(f"Error has been occured into model_creation() function in model building module & the error is {e}")
        console_logger.error(f"Error has been occured into model_creation() function in model building module & the error is {e}")
        raise CustomException(e,sys)


def main()->None:

    try:
        root = Path(__file__).parent.parent.parent
        train_data_path = root/"data"/"processed"/"processed_train.csv"

        train_df = load_data(train_data_path)

        X_train = train_df.iloc[:,:-1]
        y_train = train_df.iloc[:,-1]

        file_logger.info("training data is loaded & has been splitted into X_train,y_train")
        console_logger.info("training data is loaded & has been splitted into X_train,y_train")

        model = model_creation()

        file_logger.info("Model has been loaded")
        console_logger.info("Model has been loaded")

        final_model = model.fit(X_train, y_train)
        stacking_reg = final_model.regressor_
        transformer = final_model.transformer_

        file_logger.info("Model has been trained")
        console_logger.info("Model has been trained")

        model_path = root/"models"/"final_model.pkl"
        staking_model_path = root/"models"/"stacking_regressor.pkl"
        transformer_path = root/"models"/"stacking_transformer.pkl"

        save_model(final_model,model_path)
        file_logger.info("Final Model has been saved")
        console_logger.info("Final Model has been saved")

        save_model(stacking_reg, staking_model_path)
        file_logger.info("Stacking regressor has been saved")
        console_logger.info("Stacking regressor has been saved")

        with open(transformer_path, "wb") as file:
            pickle.dump(transformer, file)
        file_logger.info("Power transformer has been saved")
        console_logger.info("Power transformerhas been saved")
    
    except Exception as e:
        file_logger.error(f"Error has been occured into main() function in model building module & the error is {e}")
        console_logger.error(f"Error has been occured into main() function in model building module & the error is {e}")
        raise CustomException(e,sys)

#====================================================================================================
    
if __name__ == "__main__":
    #this is main function
    main()
    print("Model Building has been successfully executed.")
