import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adamax
from keras.losses import Huber

from TA_LSTMxFlutter.essential.global_params import LABEL_WIDTH

def models(train_data= None,val_data=None,test_data= None,max_epochs=100, num_features= 8):
    model = Sequential([
        LSTM(750, input_shape=[LABEL_WIDTH,num_features]),
        Dense(8)
    ])
    model.compile(
        loss= Huber(),
            optimizer=Adamax(learning_rate=5e-4),
            metrics=["mae"]
    )
    model.fit(train_data, epochs=max_epochs, validation_data= val_data,)

    return model