from src.sentimentAnalysis.constants import *
from src.sentimentAnalysis.utils.common_functionality import read_yaml, create_directories
from src.sentimentAnalysis.entity.config_entity import DataIngestionConfig, DataProcessingConfig, ModelBuildingConfig, ModelEvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
            
        )
        
        return data_ingestion_config


    def get_data_processing_config(self) -> DataProcessingConfig:
        config = self.config.data_processing

        create_directories([config.root_dir])

        data_processing_config = DataProcessingConfig(
            root_dir=config.root_dir,
            
            word2vec_modl_file=config.word2vec_modl_file,
            final_data_file = config.final_data_file
            
        )

        return data_processing_config


    def get_model_building_config(self) -> ModelBuildingConfig:
        config = self.config.model_building

        create_directories([config.root_dir])

        model_building_config = ModelBuildingConfig(
            root_dir = config.root_dir,
            base_modl_file = config.base_modl_file,
            trained_modl_file = config.trained_modl_file,
            test_data_file = config.test_data_file,
            number_of_neurons_in_first_layer = self.params.NUMBER_OF_NEURONS_IN_FIRST_LAYER,
            number_of_neurons_in_second_layer = self.params.NUMBER_OF_NEURONS_IN_SECOND_LAYER,
            number_of_neurons_in_output_layer = self.params.NUMBER_OF_NEURONS_IN_OUTPUT_LAYER,
            metrics = self.params.METRICS,
            learning_rate = self.params.LEARNING_RATE,
            epochs = self.params.EPOCHS

            
        )

        return model_building_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        model_evaluation_config = ModelEvaluationConfig(
            trained_model_path = self.config.model_building.trained_modl_file,
            test_data_path = self.config.model_building.test_data_file

            
        )

        return model_evaluation_config