# Importing Library
import os
import re
import time
import datetime


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


from Essential import path_handling as ph
from Essential import global_params as gp
from Data_Processing import data_processing as dp
from Data_Processing import training_prep as tp
from Training import models
from Plotting import models_plot as mp


def main():
    start_time= datetime.datetime.now()
    print(f'LSTMxFlutter Script called at{start_time}')

    # Initialize Dictionary
    ph.InitDataDirectories()

    #Check whether the model already trained or not?
    if os.path.isfile(ph.GetModelsData()) == False:
        # Processing Data
        dp.data_process()

        # Get dictionary of train_data
        train_dict= dp.GetDictTrainData()

        # Training
        for name, train in train_dict.items():
            print('START TRAINING')
            df_norm, params = tp.df_norm(train, minmax= True)
            train_df, val_df, _, num_features = tp.train_val_split(df_norm)
            print(f'df shape is {train.shape}')
            print(f'train_df shape is {train_df.shape}')
            print(f'val_df shape is {val_df.shape}')

            window= tp.WindowGenerator(input_width=1, label_width=1, shift=1,
                                    train_df = train_df, val_df= val_df,
                                    label_columns=None, batch_size= 40)
            train = window.train
            val = window.val

            features, labels = next(iter(train))
            print (f'train features shape is {features.shape}')
            print (f'train labels shape is {labels.shape}')

            val_features, val_labels = next(iter(val))
            print (f'val features shape is {val_features.shape}')
            print (f'val labels shape is {val_labels.shape}')
            
            model,history = models.model(train_data= train,val_data= val,test_data= None, num_features=num_features)
            models.save_model(model,names=name)
    else:
        pass 