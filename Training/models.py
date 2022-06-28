import tensorflow as tf
import numpy as np
import os
import pandas as pd


from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.losses import Huber
from Data_Processing.training_prep import df_norm

from Essential import global_params as gp
from Essential import path_handling as ph
from Data_Processing.training_prep import WindowGenerator, train_val_split



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
        LSTM(300, input_shape=[gp.LABEL_WIDTH,num_features]),
        Dense(num_features)
    ])
    model.compile(
        loss= Huber(),
        optimizer=tf.keras.optimizers.Adamax(learning_rate= 5e-4),
        metrics=["mae"]
    )
    history= model.fit(train_data, epochs=max_epochs, verbose=0,
        validation_data= val_data,callbacks= myEarlyStop)
    print('Training Done')
    return model, history

def save_model(model,names= None, no_case= None, no_model= None , vf_case=None, path= ph.GetModelsData()):
    if names == None:
        if no_case==0: 
            names=f'LSTM{no_case}{no_model}.h5'
        elif no_case==1:
            names=f'LSTM{no_case}{no_model}{vf_case}.h5'
    names= f'LSTM{names}.h5'
    dir = os.path.join(path, names)
    model.save(dir)
    print('Done Saving Model')


def load_and_evaluate(file:str =None,test_file=None,type= None, data_path= ph.GetProcessedData(),models_path=ph.GetModelsData(),target_path= ph.GetModelPerformancesData()):
    models, models_perform=dict(), dict()
    if test_file != None:
        path = os.path.join(data_path, test_file)
        if file != None:
            model_path = os.path.join(models_path, file)
            model = tf.keras.models.load_model(model_path)
            df= pd.read_csv(path)
            df, params = df_norm(df, minmax= True)
            if type != None:
                train_df, val_df, _, num_features = train_val_split(df)
                window= WindowGenerator(train_df=train_df, val_df=val_df, batch_size=40)
                train = window.train
                val = window.val

                if type=='train':
                    test = train
                elif type=='val':
                    test = val
                else:
                    print('only support type= train, val, and left it empty to combine train and val')
                
            else:
                window= WindowGenerator(train_df=df, batch_size=40)
                test = window.train
            
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
                df, params = df_norm(df, minmax=True)
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


def forecast(model=None, test_file=None,):
    pass