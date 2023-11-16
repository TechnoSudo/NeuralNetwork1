import os.path

import numpy as np
from neural_network import gradient_clip
import numpy as np
import matplotlib.pyplot as plt


class DenseLayer:
    def __init__(self, input_size, output_size, initialization_method):
        self.last_input = None
        modifier = 0.5 * (input_size * output_size) ** -0.5
        if initialization_method == "normal":
            self.weights = np.random.randn(output_size, input_size)
            self.biases = np.random.randn(output_size, 1)
        elif initialization_method == "uniform":
            self.weights = np.random.uniform(low=-1, high=1, size=(output_size, input_size))
            self.biases = np.random.uniform(low=-1, high=1, size=(output_size, 1))

    def forward(self, x):
        self.last_input = x
        return np.dot(self.weights, x) + self.biases

    def backward(self, output_gradient):
        weights_gradient = np.dot(output_gradient, self.last_input.T)

        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient, weights_gradient

    def backward_update(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.last_input.T)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient

        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient

    def update(self, weights_gradient, output_gradient, learning_rate):
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient


class ActivationLayer:
    def __init__(self, activation_function, activation_function_derivative):
        self.last_input = None
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward(self, x: np.array) -> np.array:
        self.last_input = x
        return self.activation_function(x)

    def backward(self, output_gradient: np.array) -> np.array:
        return np.multiply(output_gradient, self.activation_function_derivative(self.last_input)), None

    def update(self, *args):
        pass

    def backward_update(self, output_gradient: np.array, learning_rate) -> np.array:
        return np.multiply(output_gradient, self.activation_function_derivative(self.last_input))


class Relu(ActivationLayer):
    def __init__(self):
        def relu(x):
            return np.vectorize(lambda a: a if a > 0 else 0)(x)

        def relu_derivative(x):
            return np.vectorize(lambda a: 1 if a > 0 else 0)(x)

        super().__init__(relu, relu_derivative)


class LeakyRelu(ActivationLayer):
    def __init__(self):
        def leaky_relu(x):
            return np.vectorize(lambda a: a if a > 0 else -0.2 * a)(x)

        def leaky_relu_derivative(x):
            return np.vectorize(lambda a: 1 if a > 0 else -0.2)(x)

        super().__init__(leaky_relu, leaky_relu_derivative)


class Tanh(ActivationLayer):

    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - (np.tanh(x) ** 2)

        super().__init__(tanh, tanh_prime)


class Sigmoid(ActivationLayer):

    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(x))

        def sigmoid_derivative(x):
            sig = sigmoid(x)
            return sig * (1 - sig)

        super().__init__(sigmoid, sigmoid_derivative)

class SoftMax():
    def forward(self, x):
        # for more stable performance:
        # normalized_x = x - np.max(x)
        x_exp = np.exp(x)
        self.last_output = x_exp / np.sum(x_exp)
        return self.last_output

    def softmax_derivative(self, x):
        normalized_last_input = self.last_input - np.max(self.last_input)
        summation = np.sum(np.exp(normalized_last_input))
        base = self.last_output / np.sum(summation)

        base *= (- np.exp(self.last_input) + summation)
        return gradient_clip(base, 10)
        return np.interp(base, (base.min(), base.max()), (-10, +10))

    def backward(self, output_gradient):
        n = np.size(self.last_output)
        return np.dot((np.identity(n) - self.last_output.T) * self.last_output, output_gradient), None

    def backward_update(self, output_gradient, learning_rate):
        n = np.size(self.last_output)
        return np.dot((np.identity(n) - self.last_output.T) * self.last_output, output_gradient)

    def another_optimization(self, x):
        s = self.forward(x)
        return s * (1 - s)

    def update(self, *args):
        pass


def save_hidden_layer_data(input_layer: DenseLayer, folder_name, sub_name):
    squarable = False
    shape = input_layer.weights.shape[1]
    if int(shape**0.5)**2 == shape:
        squarable = True


    if not os.path.isdir(folder_name): os.mkdir(folder_name)
    if not os.path.isdir(folder_name + "/latent_features"): os.mkdir(folder_name + "/latent_features")
    if not os.path.isdir(folder_name + "/latent_features/"+sub_name): os.mkdir(folder_name + "/latent_features/"+sub_name)

    image_lists = input_layer.weights

    # Function to reshape and plot images with positive values in green and negative values in red
    def plot_images_with_color(image_lists):
        for i, image_data in enumerate(image_lists):
            # Reshape the flattened image to 28x28
            if squarable:
                image = np.reshape(image_data, (int(shape**0.5),int(shape**0.5)))
                # Initialize an RGB image with zeros
                rgb_image = np.zeros((int(shape**0.5), int(shape**0.5), 3), dtype=np.uint8)
            else:
                image = np.reshape(image_data, (shape))
                # Initialize an RGB image with zeros
                rgb_image = np.zeros((shape, 3), dtype=np.uint8)


            # Set positive values to green (0, 255, 0)
            rgb_image[image > 0, 1] = 255

            # Set negative values to red (255, 0, 0)
            rgb_image[image < 0, 0] = 255

            # Plot the image
            plt.imshow(rgb_image)
            plt.title(f'Neuron {i + 1}')
            plt.savefig(f"{folder_name}/latent_features/{sub_name}/Neuron_{i + 1}.png")

    # Plot the images with color mapping
    plot_images_with_color(image_lists)
    plt.close()


def save_error_graph(error_per_epoch, working_dir):
    plt.plot(np.linspace(1, len(error_per_epoch), len(error_per_epoch)), error_per_epoch, label="Error graph")
    plt.title('Error per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.savefig(f"{working_dir}/error_graph.png")
    plt.close()

def save_hyperparameter_graph(data: list, working_dir):
    hidden_sizes, accuracies = zip(*data)
    plt.bar(hidden_sizes, accuracies)
    plt.title('Accuracy per hidden layer size')
    plt.xlabel('Hidden layer size')
    plt.ylabel('Accuracy')
    plt.savefig(f"{working_dir}/accuracy_hidden_size_graph.png")
    plt.close()