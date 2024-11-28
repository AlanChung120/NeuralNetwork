# version 1.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from operations import *

def load_dataset(csv_path, target_feature):
    dataset = pd.read_csv(csv_path)
    t = np.expand_dims(dataset[target_feature].to_numpy().astype(float), axis=1)
    X = dataset.drop([target_feature], axis=1).to_numpy()
    return X, t

X, y = load_dataset("data/wine_quality.csv", "quality")

n_features = X.shape[1]
epochs = 10

k = 5
epoch_losses = []
mae_list = []
for i in range(k):
  xLen = X.shape[0]
  yLen = y.shape[0]
  x_test_size = xLen / k
  y_test_size = yLen / k

  X_test = np.array([X[index] for index in range(xLen) if (((i / k) * xLen) <= index and index < ((i / k) * xLen + x_test_size))])
  y_test = np.array([y[index] for index in range(yLen) if (((i / k) * yLen) <= index and index < ((i / k) * yLen + y_test_size))])
  X_train = np.array([X[index] for index in range(xLen) if (((i / k) * xLen) > index or index >= ((i / k) * xLen + x_test_size))])
  y_train = np.array([y[index] for index in range(yLen) if (((i / k) * yLen) > index or index >= ((i / k) * yLen + y_test_size))])

  net = NeuralNetwork(n_features, [32,32,16,1], [ReLU(), ReLU(), Sigmoid(), Identity()], MeanSquaredError(), learning_rate=0.001)
  trained_W, epoch_loss = net.train(X_train, y_train, epochs)
  epoch_losses.append(epoch_loss)
  mae_list.append(net.evaluate(X_test, y_test, mean_absolute_error))


epoch_losses_avg = []
for epochIndex in range(epochs):
  epochAvg = 0
  for i in range(k):
    epochAvg += epoch_losses[i][epochIndex]
  epochAvg /= k
  epoch_losses_avg.append(epochAvg)
std_mae = np.std(mae_list)
mean_mae = np.mean(mae_list)

print(f"Standard Deviation of MAE: {std_mae}, Mean of MAE: {mean_mae}")
plt.plot(np.arange(0, epochs), epoch_losses_avg)
plt.title('Average Epoch training loss over 5 runs for each Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Epoch training loss over 5 runs')
plt.show()
