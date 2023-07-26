from src.sentimentAnalysis.utils.common_functionality import get_size, save_object, load_object, sent_to_vector, text_preprocessing
from src.sentimentAnalysis import logger
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import numpy as np
import os
from src.sentimentAnalysis.entity.config_entity import DataProcessingConfig


class DataProcessing:
    def __init__(self, config: DataProcessingConfig):
        self.config = config


    
    def process_text(self,data_path):
        if not os.path.exists(self.config.final_data_file):

            
            df = pd.read_csv(data_path)
            df = df.dropna(axis=0)

            df = df.drop_duplicates()
            word2vec_model = load_object(self.config.word2vec_modl_file)

            df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
            df['review'] = df['review'].apply(lambda x: text_preprocessing(x))
            df['review'] = df['review'].apply(lambda x: simple_preprocess(x))
            df['vec'] = df['review'].apply(lambda x: sent_to_vector(word2vec_model,x))


            X = np.array(df['vec'].tolist())
            y = np.array(df['sentiment'].tolist())

            final_data_dict = {
                'X':X,
                'y':y
            }
            logger.info("Final data created")

            save_object(self.config.final_data_file,final_data_dict)
            logger.info("Final data saved")

        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.final_data_file))}")




    def train_word2vec_model(self,data_path):

        if not os.path.exists(self.config.word2vec_modl_file):

            df = pd.read_csv(data_path)
            df = df.dropna(axis=0)

            df = df.drop_duplicates()

            reviewText = df['review'].apply(lambda x: simple_preprocess(x))
            model = gensim.models.Word2Vec(
                window = 10,
                min_count = 2,
                workers = 4
            )

            model.build_vocab(reviewText, progress_per = 100)
            model.train(reviewText, total_examples = model.corpus_count, epochs = model.epochs)
            logger.info("word2vec model training completed")

            save_object(self.config.word2vec_modl_file, model)
            logger.info("Word2Vec model is saved")

        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.word2vec_modl_file))}")
