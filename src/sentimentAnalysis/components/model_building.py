from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from src.sentimentAnalysis.utils.common_functionality import load_object, save_object, get_size
from src.sentimentAnalysis import logger
from src.sentimentAnalysis.entity.config_entity import ModelBuildingConfig
import os
from tensorflow.keras.models import load_model as tfk__load_model



class ModelBuilding:
    def __init__(self, config: ModelBuildingConfig):
        self.config = config


    
    def prepare_base_model(self):
        if not os.path.exists(self.config.base_modl_file):
            model = Sequential()

            model.add(LSTM(self.config.number_of_neurons_in_first_layer, input_shape = (1,100), return_sequences = True))
            model.add(LSTM(self.config.number_of_neurons_in_second_layer))
            model.add(Dense(self.config.number_of_neurons_in_output_layer, activation = 'sigmoid'))

            print(model.summary())

            loss = keras.losses.BinaryCrossentropy(from_logits = True)
            opt = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            metrices = [self.config.metrics]

            model.compile(loss=loss,optimizer=opt,metrics=metrices)
            logger.info("Model comiplation is done")

            #save_object(self.config.base_modl_file,model)
            model.save(self.config.base_modl_file)
            logger.info("Base model is saved")


        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")



    def train_model(self,final_data_path):
        if not os.path.exists(self.config.trained_modl_file):
            
            final_data_for_training = load_object(final_data_path)

            X = final_data_for_training['X']
            y = final_data_for_training['y']

            X = X.reshape(X.shape[0],1,X.shape[1])
            y = y.reshape(-1,1)

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

            #base_model = load_object(self.config.base_modl_file)
            base_model = tfk__load_model(self.config.base_modl_file)

            base_model.fit(x=X_train,y=y_train,epochs=self.config.epochs)
            logger.info("Model training completed")

            #save_object(self.config.trained_modl_file, base_model)
            base_model.save(self.config.trained_modl_file)
            logger.info("trained model is saved")

            test_data = {
                'X_test': X_test,
                'y_test': y_test
            }

            save_object(self.config.test_data_file, test_data)
            logger.info("test data saved for evaluation phase")

        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
            



