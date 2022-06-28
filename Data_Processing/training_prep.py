
# This Window Generator Code was snipped from https://www.tensorflow.org/tutorials/structured_data/time_series


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from TA_LSTMxFlutter.Essential import global_params as gp

# Create Sequences dataset
class WindowGenerator():
  def __init__(self, input_width=gp.INPUT_WIDTH, label_width=gp.LABEL_WIDTH, shift=gp.SHIFT,
               train_df=None, val_df=None, test_df=None,
               label_columns=None, batch_size= 32):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.batch_size = batch_size

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

      # Slicing doesn't preserve static shape information, so set the shapes
      # manually. This way the `tf.data.Datasets` are easier to inspect.
      inputs.set_shape([None, self.input_width, None])
      labels.set_shape([None, self.label_width, None])

      return inputs, labels

  def make_dataset(self, data, batch_size= 32):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.utils.timeseries_dataset_from_array(
          data=data,
          targets=None,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=False,
          batch_size=self.batch_size,)

      ds = ds.map(self.split_window)

      return ds  
  
  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  @property
  def train(self):
      return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
   return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
    # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
    # And cache it for next time
      self._example = result
    return result


# Split Dataset to train, and validation
def train_val_split(dataframe, train_split= 0.8, test_portion= None):
    column_indices = {name: i for i, name in enumerate(dataframe.columns)}

    n = len(dataframe)
    train_df = dataframe[0:int(n*train_split)]
    val_df = dataframe[int(n*train_split):]
    test_df = None
    if test_portion is not None:
        test_split = (1-test_portion)*n
        val_df= dataframe[int(n*train_split):test_split]
        test_df= dataframe[test_split:]

    num_features = dataframe.shape[1]
    return train_df, val_df, test_df, num_features





# Normalize Data
def norm(data, min, max, mean, std, minmax= False):
    if minmax == True:
        return (data-min)/(max-min)
    else:
        return (data-mean)/std

def denorm(data, min, max, mean, std, minmax= False):
    if minmax == True:
        return data*(max-min) + min
    else:
        return (data * std) + mean
    
def df_norm(dataframe, minmax= False):
    columns= dataframe.columns
    col_params = {}
    for column in columns:
        min = dataframe[column].min()
        max = dataframe[column].max()
        mean = dataframe[column].mean()
        std = dataframe[column].std()
        col_params[column] = min, max, mean, std
        if minmax == True:  
            dataframe[column] = norm(dataframe[column], min, max, mean, std, minmax= True)
        else:
            dataframe[column] = norm(dataframe[column], min, max, mean, std, minmax= False)
    return dataframe, col_params

def df_denorm(dataframe,col_params, minmax= False):
    columns = dataframe.columns
    dict_keys = col_params.keys()
    for column, key in zip(columns, dict_keys):
        if column == key:
            if minmax == True:
                dataframe[column] = denorm(dataframe[column], col_params[column][0], col_params[column][1],
                                           col_params[column][2], col_params[column][3], minmax= True)
            dataframe[column] = denorm(dataframe[column], col_params[column][0], col_params[column][1],
                                           col_params[column][2], col_params[column][3], minmax= False)
    return dataframe