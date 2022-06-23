import tensorflow as tf
import numpy as np
import os

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.losses import Huber

from TA_LSTMxFlutter.essential import global_params as gp
from TA_LSTMxFlutter.essential import path_handling as ph


def model(train_data= None,val_data=None,test_data= None,max_epochs=100, num_features= 8):
    myEarlyStop= tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-5,
    patience=20,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

    model = Sequential([
        LSTM(750, input_shape=[gp.LABEL_WIDTH,num_features]),
        Dense(8)
    ])
    model.compile(
        loss= Huber(),
            optimizer=tf.keras.optimizers.Adamax(learning_rate= 5e-4),
            metrics=["mae"]
    )
    model.fit(train_data, epochs=max_epochs, validation_data= val_data,callbacks= myEarlyStop)

    return model

def save_model(model,no_case= None, no_model= None , vf_case=None, path= ph.GetModelsData()):
    if no_case==0: 
        names=f'LSTM{no_case}{no_model}.h5'
    elif no_case==1:
        names=f'LSTM{no_case}{no_model}{vf_case}.h5'
    dir = os.path.join(path, names)
    model.save(dir)