from src.sentimentAnalysis.utils.common_functionality import load_object, text_preprocessing, sent_to_vector
from gensim.utils import simple_preprocess
import os
from tensorflow.keras.models import load_model as tfk__load_model


class PredictionPipeline:
    def __init__(self, text):
        self.text = text


    def predict(self):
        #model = load_object(os.path.join("artifacts","model_building","trained_model.pkl"))
        model = tfk__load_model(os.path.join("artifacts","model_building","model.h5"))
        word2vec = load_object(os.path.join("artifacts","data_processing","word2vec.pkl"))
        preprocessed_text = text_preprocessing(self.text)
        list_of_tokens = simple_preprocess(preprocessed_text)
        vector = sent_to_vector(word2vec,list_of_tokens)

        vector = vector.reshape(1,1,len(vector))

        predicted_value = model.predict(vector)[0][0]



        return predicted_value*100

