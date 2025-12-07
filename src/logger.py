import logging
import os
from datetime import datetime



def console_logging(name):

    logger=logging.getLogger(name)
    logger.setLevel("DEBUG")

    console_handler=logging.StreamHandler()
    console_handler.setLevel("DEBUG")

    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

    

def file_logging(name):

    LOG_FOLDER=f"{datetime.now().strftime('%m_%d_%Y')}"
    LOG_FILE=f"{datetime.now().strftime('%H_%M_%S')}.log"
    logs_path=os.path.join(os.getcwd(),"logs",LOG_FOLDER)
    os.makedirs(logs_path,exist_ok=True)
    LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

    logger=logging.getLogger(name)
    logger.setLevel("DEBUG")

    file_handler=logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel("DEBUG")

    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger