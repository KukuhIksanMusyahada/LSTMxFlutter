import matplotlib.pyplot as plt


from TA_LSTMxFlutter.essential import path_handling as ph

def history_plot(history, path = ph.GetModelPerformancesData()):
  '''Plots the training and validation loss and mae from a history object'''
  mae = history.history['mae']
  val_mae = history.history['val_mae']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  titles= ['Training and validation mae', 'Training and validation loss']
  epochs = range(len(mae))

  plt.plot(epochs, mae, 'bo', label='Training Mean Absolute Error')
  plt.plot(epochs, val_mae, 'b', label='Validation Mean Absolute Error')
  plt.title(titles[0])

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training Huber Loss')
  plt.plot(epochs, val_loss, 'b', label='Validation Huber Loss')
  plt.title(titles[1])
  plt.legend()

  plt.show()
  plt.savefig('.')

def model_forecast_plot():
    pass