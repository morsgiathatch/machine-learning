import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def map_estimator_gradient(x, y, args):
    # Batch gradient
    w = args[0]
    var = args[1]
    return ml_estimator_gradient(x, y, [w]) + (1.0 / var) * w


def map_estimator(x, y, args):
    w = args[0]
    var = args[1]
    return ml_estimator(x, y, [w]) + (0.5 / var) * w.dot(w)


def ml_estimator_gradient(x, y, args):
    # Batch gradient
    w = args[0]
    return -y * (1.0 - sigmoid(y * w.dot(x))) * x


def ml_estimator(x, y, args):
    w = args[0]
    _sum = 0.0
    for i in range(0, y.shape[0]):
        _sum += np.log(1.0 + np.exp(-y[i] * w.dot(x[i, :])))

    return _sum


def predictor(x, w):
    return np.sign(w.dot(x))


def get_percentages(data, weights):
    num_correct = 0
    for row_ndx in range(0, data.features.shape[0]):
        if data.output[row_ndx] == predictor(data.features[row_ndx, :], weights):
            num_correct += 1

    return 1.0 - float(num_correct / data.features.shape[0])
