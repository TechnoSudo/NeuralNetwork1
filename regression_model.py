import random

import numpy as np

import loss_functions
from layers import *
from loss_functions import *
from neural_network import *
import matplotlib.pyplot as plt

# Generate data points lying on the circle
# x^2 + y^2 = 16

x = np.load("data/processed/factorsCancer.npy")
data_size = x.shape[0]
x = x.reshape(data_size, x.shape[1], 1)
y = np.load("data/processed/labelsCancer.npy").reshape(data_size, 1)

zipped = list(zip(x, y))
random.shuffle(zipped)
x = np.array([a[0] for a in zipped])
y = np.array([a[1] for a in zipped])

x_train, x_validate, x_test = split_data(x, [0.8, 0.1, 0.1])
y_train, y_validate, y_test = split_data(y, [0.8, 0.1, 0.1])

working_dir = "latest_model"
if not os.path.isdir(working_dir): os.mkdir(working_dir)

initialization_methods = ["normal", "uniform"]

accuracy = 0
# gridSearch

for hidden_size in range(4, 17, 4):
    for learning_rate in np.linspace(0.01, 1, 4):
        for min_batch_size in [8, 16, 32]:
            for initialization_method in initialization_methods:
                regression_model = NeuralNetwork(
                    DenseLayer(23, hidden_size, initialization_method),
                    Tanh(),
                    DenseLayer(hidden_size, 1, initialization_method),
                    Tanh()
                )

                regression_model.mini_batch_training(loss_functions.mean_squared_error,
                                     loss_functions.mean_squared_error_derivative,
                                     x_train, y_train, epochs=20, learning_rate=learning_rate,
                                     mini_batch_size=32, randomize=True, verbose=False)
                accuracy = regression_model.get_accuracy(x_validate, y_validate)
                print(f"accuracy: {accuracy} , hidden size:{hidden_size},learning_rate: {learning_rate}, mini_batch_size {min_batch_size}, initialization_method {initialization_method}")
                # if accuracy > 0.99:
                #     exit()

save_hidden_layer_data(regression_model.layers[0],"classification", "first_layer")
save_hidden_layer_data(regression_model.layers[2],"classification", "second_layer")





