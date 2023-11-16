import random

import numpy as np

import loss_functions
from layers import *
from loss_functions import *
from neural_network import *
import matplotlib.pyplot as plt


X = np.load("data/processed/pixelsNumbers.npy")
data_size = X.shape[0]
X = X.reshape(data_size, X.shape[1], 1)
Y = np.load("data/processed/labelsNumbers.npy").reshape(data_size, 10, 1)

# optional preprocessing
X = np.array([[[a / 255] for a in item] for item in X]).reshape(data_size, 784, 1)

x_train, x_validate, x_test = split_data(X, [0.8, 0.1, 0.1])
y_train, y_validate, y_test = split_data(Y, [0.8, 0.1, 0.1])

working_dir = "classification"
if not os.path.isdir(working_dir): os.mkdir(working_dir)

initialization_methods = ["normal", "uniform"]

accuracy = 0
# gridSearch
# for hidden_size in range(16, 65, 8):
#     for learning_rate in np.linspace(0.01, 0.2, 10):
#         for min_batch_size in [8, 16, 32]:
#             for initialization_method in initialization_methods:


accuracy_list = []

for hidden_size in range(16, 65, 16):

    classification_model = NeuralNetwork(
        DenseLayer(784, hidden_size, initialization_methods[1]),
        Tanh(),
        DenseLayer(hidden_size, 10, initialization_methods[1]),
        SoftMax()
    )

    error_data = classification_model.mini_batch_training(loss_functions.categorical_crossentropy,
                         loss_functions.categorical_crossentropy_derivative,
                         x_train, y_train, epochs=50, learning_rate=0.01,
                         mini_batch_size=32, randomize=True, verbose=True)
    accuracy = classification_model.get_accuracy_argmax(x_validate, y_validate)
    accuracy_list.append((hidden_size, accuracy))
    print(f"{accuracy}")

    save_hidden_layer_data(classification_model.layers[0],working_dir, "first_layer")
    save_hidden_layer_data(classification_model.layers[2],working_dir, "second_layer")
    save_error_graph(error_data, working_dir)

save_hyperparameter_graph(accuracy_list, working_dir)

