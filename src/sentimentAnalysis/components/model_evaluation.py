from src.sentimentAnalysis.utils.common_functionality import load_object,save_json
from src.sentimentAnalysis import logger
from src.sentimentAnalysis.entity.config_entity import ModelEvaluationConfig
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    
    def evaluate(self, model_path, test_data_path):
        
        trained_model = load_object(model_path)
        test_data = load_object(test_data_path)
        X_test = test_data['X_test']
        y_test = test_data['y_test']

        scores = trained_model.evaluate(X_test, y_test)
        score_dic={
            'loss': scores[0],
            'accuracy': scores[1]
        }

        save_json(Path('scores.json'),score_dic)
        logger.info("Model score is saved")
          


