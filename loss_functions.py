import numpy as np

def categorical_crossentropy(y_true: np.array, y_pred: np.array):
    bound_limit = 0.000001
    # approximation for log function:
    y_pred = np.array([[bound_limit] if x < bound_limit else ([1-bound_limit] if x > bound_limit else [x]) for x in y_pred])

    return - np.mean(y_true * np.log(y_pred))

def categorical_crossentropy_derivative(y_true: np.array, y_pred: np.array):
    bound_limit = 0.000001
    # approximation for log function:
    y_pred = np.array([[bound_limit] if x < bound_limit else ([1-bound_limit] if x > bound_limit else [x]) for x in y_pred])

    return - y_true / y_pred


def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
