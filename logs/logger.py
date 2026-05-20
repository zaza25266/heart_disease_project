
import logging
import os
from datetime import datetime

# check if los directory exists, if not create it
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_core_logger(module_name: str):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # prevent duplicates
    if not logger.handlers:
        # create a daily log file
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file_path = os.path.join(LOG_DIR, f"{current_date}.log")
        
        # setup handlers (file and console)
        file_handler = logging.FileHandler(log_file_path)
        console_handler = logging.StreamHandler()
        
        # log format
        log_format = logging.Formatter(
            '%(asctime)s | [%(levelname)s] | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # attach handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)  
        
    return logger