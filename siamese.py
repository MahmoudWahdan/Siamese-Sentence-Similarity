# -*- coding: utf-8 -*-
"""
Created on Thu May 24 00:27:33 2018

@author: mwahdan
"""

from keras import layers
from keras import Input
from keras.models import Model, model_from_json
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import keras.backend as K
import matplotlib.pyplot as plt
from metrics import pearson_correlation

class SiameseModel:
    
    def __init__(self):
        n_hidden = 50
        input_dim = 300
        
        # Use CuDNNLSTM instead of LSTM, because it is faster
        # unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force  bias_initializer="zeros". This is recommended in Jozefowicz et al.
        # he_normal: Gaussian initialization scaled by fan_in (He et al., 2014)
        lstm = layers.CuDNNLSTM(n_hidden, unit_forget_bias=True, 
                                kernel_initializer='he_normal', 
                                kernel_regularizer='l2',
                                name='lstm_layer')
#        lstm = layers.LSTM(n_hidden, unit_forget_bias=True, kernel_initializer='he_normal')
        
        # Building the left branch of the model: inputs are variable-length sequences of vectors of size 128.
        left_input = Input(shape=(None, input_dim), name='input_1')
#        left_masked_input = layers.Masking(mask_value=0)(left_input)
        left_output = lstm(left_input)
        
        # Building the right branch of the model: when you call an existing layer instance, you reuse its weights.
        right_input = Input(shape=(None, input_dim), name='input_2')
#        right_masked_input = layers.Masking(mask_value=0)(right_input)
        right_output = lstm(right_input)
        
        # Builds the classifier on top
        l1_norm = lambda x: 1 - K.abs(x[0] - x[1])
        merged = layers.merge([left_output, right_output], mode = l1_norm, 
                              output_shape = lambda x: x[0],
                              name='L1_distance')
        predictions = layers.Dense(1, activation='sigmoid', name='Similarity_layer')(merged)
        
        # Instantiating and training the model: when you train such a model, the weights of the LSTM layer are updated based on both inputs.
        self.model = Model([left_input, right_input], predictions)
        
        self.__compile()
        print(self.model.summary())
        # plot graph
        plot_model(self.model, to_file='siamese_architecture.png')
        
    def __compile(self):
        optimizer = Adadelta() # gradient clipping is not there in Adadelta implementation in keras
#        optimizer = 'adam'
        self.model.compile(loss = 'mse', optimizer = optimizer, metrics=[pearson_correlation])
        
    def fit(self, left_data, right_data, targets, validation_data, epochs=5, batch_size=128):
        # The paper employ early stopping based on a validation, but they didn't mention parameters.
        early_stopping_monitor = EarlyStopping(monitor = 'val_pearson_correlation', mode='max', patience = 20)
#        callbacks = [early_stopping_monitor]
        callbacks = []
        history = self.model.fit([left_data, right_data], targets, validation_data=validation_data,
                                 epochs = epochs, batch_size = batch_size#)
                                 , callbacks = callbacks)
        
        self.visualize_metric(history.history, 'loss')
        self.visualize_metric(history.history, 'pearson_correlation')
        
    def visualize_metric(self, history_dic, metric_name):
        plt.plot(history_dic[metric_name])
        legend = ['train']
        if 'val_' + metric_name in history_dic:
            plt.plot(history_dic['val_' + metric_name])
            legend.append('test')
        plt.title('model ' + metric_name)
        plt.ylabel(metric_name)
        plt.xlabel('epoch')
        plt.legend(legend, loc='upper left')
        plt.show()
        
    def predict(self, left_data, right_data):
        return self.model.predict([left_data, right_data])
    
    def evaluate(self, left_data, right_data, targets, batch_size=128):
        return self.model.evaluate([left_data, right_data], targets, batch_size=batch_size)
    
    def save(self, model_folder='./model/'):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_folder + 'model.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_folder + 'model.h5')
        print('Saved model to disk')
        
    def save_pretrained_weights(self, model_wieghts_path='./model/pretrained_weights.h5'):
        self.model.save_weights(model_wieghts_path)
        print('Saved pretrained weights to disk')
        
    def load(self, model_folder='./model/'):
        # load json and create model
        json_file = open(model_folder + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_folder + 'model.h5')
        print('Loaded model from disk')
        
        self.model = loaded_model
        # loaded model should be compiled
        self.__compile()
        
    def load_pretrained_weights(self, model_wieghts_path='./model/pretrained_weights.h5'):
        # load weights into new model
        self.model.load_weights(model_wieghts_path)
        print('Loaded pretrained weights from disk')
        self.__compile()
        