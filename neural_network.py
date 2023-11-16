import random
from layers import *
import numpy as np

def get_averages(array, split):
    new_array = []
    i = 0
    while i < split:
        new_array.append(array[i])
        i += 1
    while i < len(array):
        i += 1
        if array[i][0] is None:
            i+=1
            continue
        new_array[i % split][0] += array[i][0]
        new_array[i % split][1] += array[i][1]
    for i in range(split):
        if array[i][0] is None:
            continue
        new_array[i][0] /= (split-1)
        new_array[i][1] /= (split-1)

    return new_array[:-1]


def gradient_clip(x, bound):
    return np.array([[-bound if a < -bound else (
        bound if a > bound else a) for a in sub_array] for sub_array in x])


class NeuralNetwork:

    def __init__(self, *layers, repeat_limit=3):
        self.layers = layers
        self.repeat_limit = repeat_limit

    def predict(self, input: np.array) -> np.array:
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def mini_batch_training(self, loss_function, loss_function_derivative, x_train, y_train, learning_rate, epochs,
                            mini_batch_size, randomize=True, verbose=True):
        zipped_data = list(zip(x_train, y_train))
        errors = []
        for epoch in range(epochs):
            mini_batch_counter = 0
            layers_count = len(self.layers)
            layers_updates = []
            error = 0
            last_error = 0
            if randomize:
                random.shuffle(zipped_data)

            for x, y in zipped_data:
                mini_batch_counter += 1

                output = self.predict(x)
                error += loss_function(y, output)

                gradient = loss_function_derivative(y, output)
                layers_updates.append([None, gradient])
                for layer in reversed(self.layers):
                    gradient, weights_gradient = layer.backward(gradient)
                    layers_updates[-1][0] = weights_gradient
                    layers_updates.append([None, gradient])

                if mini_batch_counter == mini_batch_size:
                    averages = get_averages(layers_updates, layers_count+1)
                    for i in range(len(self.layers)):
                        self.layers[i].update(*averages[-1-i], learning_rate)

                    mini_batch_counter = 0
                    layers_updates = []
            if mini_batch_counter>0:
                averages = get_averages(layers_updates, layers_count + 1)
                for i in range(len(self.layers)):
                    self.layers[i].update(*averages[-1 - i], learning_rate)

            error = np.average(error)
            errors.append(error)
            if error == last_error:
                repeat_count += 1
            else:
                repeat_count = 0
            if repeat_count >= self.repeat_limit:
                break


            if verbose:
                print(f"epoch: {epoch + 1}/{epochs}, error={error}")

        return errors

    def train(self, loss_function, loss_function_derivative, x_train, y_train, epochs=10, learning_rate=0.01,
              verbose=True):
        last_error = 0
        repeat_count = 0
        for epoch in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):

                output = self.predict(x)
                error += loss_function(y, output)
                gradient = loss_function_derivative(y, output)

                for layer in reversed(self.layers):
                    gradient = layer.backward_update(gradient, learning_rate)

            error = np.average(error)
            if error == last_error:
                repeat_count += 1
            else:
                repeat_count = 0
            last_error = error
            if verbose:
                print(f"{epoch + 1}/{epochs}, error={error}")

            if repeat_count >= self.repeat_limit:
                break


    def get_accuracy_argmax(self, x_validate, y_validate):
        size = x_validate.shape[0]
        correct = 0
        for i in range(size):
            if np.argmax(self.predict(x_validate[i])) == np.argmax(y_validate[i]):
                correct += 1

        return correct/size

    def get_accuracy(self, x_validate, y_validate):
        size = x_validate.shape[0]
        correct = 0
        for i in range(size):
            if abs(self.predict(x_validate[i]) - y_validate[i]) < 0.5:
                correct += 1

        return correct/size

def split_data(x:np.array, split:list):
    data_size = x.shape[0]
    return x[:int(data_size * split[0])], x[:int(data_size * split[1])],x[:int(data_size * split[2])]
