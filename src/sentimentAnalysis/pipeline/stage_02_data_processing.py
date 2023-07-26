from src.sentimentAnalysis.config.configuration import ConfigurationManager
from src.sentimentAnalysis.components.data_processing import DataProcessing
from src.sentimentAnalysis import logger


STAGE_NAME = "Data Processing stage"

class DataProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_processing_config = config.get_data_processing_config()
        data_processing = DataProcessing(config=data_processing_config)
        data_processing.train_word2vec_model(config.config.data_ingestion.local_data_file)
        data_processing.process_text(config.config.data_ingestion.local_data_file)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataProcessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e



