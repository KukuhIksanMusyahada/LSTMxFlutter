import os
import matplotlib.pyplot as plt


from Essential import path_handling as ph

def history_plot(history,model_names=None, path = ph.GetModelPerformancesData()):
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
  path= os.path.join(path, model_names)
  plt.savefig(path)

def model_forecast_plot():
    pass