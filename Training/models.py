import tensorflow as tf
import numpy as np
import os
import pandas as pd


from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.losses import Huber
from TA_LSTMxFlutter.Data_Processing.training_prep import df_norm

from TA_LSTMxFlutter.essential import global_params as gp
from TA_LSTMxFlutter.essential import path_handling as ph
from TA_LSTMxFlutter.Data_Processing.training_prep import WindowGenerator


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

def load_and_evaluate(file:str =None,test_file=None, data_path= ph.GetProcessedData(),models_path=ph.GetModelsData(),target_path= ph.GetModelPerformancesData()):
    models, models_perform=dict(), dict()
    if test_file != None:
        path = os.path.join(data_path, test_file)
        if file != None:
            model_path = os.path.join(models_path, file)
            df= pd.read_csv(path)
            df, params = df_norm(df)
            window= WindowGenerator(train_df=df)
            test = window.train
            model = tf.keras.models.load_model(model_path)
            print(model.summary())
            print('---------------------------------------------------------------------------')
            loss, mae  = model.evaluate(test, verbose=0)
            print(f'Model`s loss: {loss} \n Model`s MAE: {mae}')
            models_perform['loss']=loss
            models_perform['mae'] = mae
            return model, models_perform
        else:
            total_file= len(os.listdir(models_path))
            print(f'Total Models detected: {total_file}')
            print('Loading Models')
            for file in os.listdir(models_path):
                model_path = os.path.join(models_path, file)
                df= pd.read_csv(path)
                df, params = df_norm(df)
                window= WindowGenerator(train_df=df)
                test = window.train
                model = tf.keras.models.load_model(model_path)
                models[file]= model
                print('---------------------------------------------------------------------------')
                loss, mae  = model.evaluate(test, verbose=0)
                models_perform[file]=[loss, mae]
            return models, models_perform
    else:
        for data in os.listdir(data_path):
            path = os.path.join(data_path, data)
            if file != None:
                model_path = os.path.join(models_path, file)
                df= pd.read_csv(path)
                df, params = df_norm(df)
                window= WindowGenerator(train_df=df)
                test = window.train
                model = tf.keras.models.load_model(model_path)
                print(model.summary())
                print('---------------------------------------------------------------------------')
                loss, mae  = model.evaluate(test, verbose=0)
                print(f'Model`s loss: {loss} \nModel`s MAE: {mae}')
                models_perform['loss']=loss
                models_perform['mae'] = mae
                return model, models_perform
            else:
                total_file= len(os.listdir(models_path))
                print(f'Total Models detected: {total_file}')
                print('Loading Models')
                for file in os.listdir(models_path):
                    model_path = os.path.join(models_path, file)
                    df= pd.read_csv(path)
                    df, params = df_norm(df)
                    window= WindowGenerator(train_df=df)
                    test = window.train
                    model = tf.keras.models.load_model(model_path)
                    models[file]= model
                    print('---------------------------------------------------------------------------')
                    loss, mae  = model.evaluate(test, verbose=0)
                    models_perform[file]=[loss, mae]
                return models, models_perform




