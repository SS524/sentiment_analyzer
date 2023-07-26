from src.sentimentAnalysis.utils.common_functionality import get_size
from src.sentimentAnalysis import logger
from src.sentimentAnalysis.entity.config_entity import DataIngestionConfig
import pandas as pd
import os



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            
            url = self.config.source_URL
            export_path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
            df = pd.read_csv(export_path)
            df.to_csv(self.config.local_data_file,index=False)
            
            logger.info("download completed!")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
