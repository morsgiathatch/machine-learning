import numpy as np


# Objective and gradient functions to be passed to gradient descent
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def map_estimator_gradient(x, y, args):
    """
    MAP estimator gradient
    :param x: features
    :type x: numpy array
    :param y: labels
    :type y: numpy array
    :param args: list in the form [weights, variance estimate]. Weights is a numpy array, variance estimate is a float
    :type args: python list
    :return: MAP estimator gradient value
    :rtype: float
    """
    # Batch gradient
    w = args[0]
    var = args[1]
    return ml_estimator_gradient(x, y, [w]) + (1.0 / var) * w


def map_estimator(x, y, args):
    """
    MAP estimator objective function
    :param x: features
    :type x: numpy array
    :param y: labels
    :type y: numpy array
    :param args: list in the form [weights, variance estimate]. Weights is a numpy array, variance estimate is a float
    :type args: python list
    :return: MAP estimator objective function value
    :rtype: float
    """
    w = args[0]
    var = args[1]
    return ml_estimator(x, y, [w]) + (0.5 / var) * w.dot(w)


def ml_estimator_gradient(x, y, args):
    """
    ML estimator gradient
    :param x: features
    :type x: numpy array
    :param y: labels
    :type y: numpy array
    :param args: list in the form [weights]. Weights is a numpy array
    :type args: python list
    :return: ML estimator gradient value
    :rtype: numpy array
    """
    # Batch gradient
    w = args[0]
    return -y * (1.0 - sigmoid(y * w.dot(x))) * x


def ml_estimator(x, y, args):
    """
    ML estimator objective function
    :param x: features
    :type x: numpy array
    :param y: labels
    :type y: numpy array
    :param args: list in the form [weights]. Weights is a numpy array
    :type args: python list
    :return: ML estimator objective value
    :rtype: float
    """
    w = args[0]
    _sum = 0.0
    for i in range(0, y.shape[0]):
        _sum += np.log(1.0 + np.exp(-y[i] * w.dot(x[i, :])))

    return _sum


def predict(x, w):
    """
    Predict feature label
    :param x: feature
    :type x: numpy array
    :param w: weights
    :param w: numpy array
    :return: +/- 1.0
    :rtype: float
    """
    return np.sign(w.dot(x))


def get_percentages(data, weights):
    """
    get error proportion
    :param data: set of features
    :type data: numpy array
    :param weights: weights
    :type weights: numpy array
    :return: error proportion
    :rtype: float
    """
    num_correct = 0
    for row_ndx in range(0, data.features.shape[0]):
        if data.output[row_ndx] == predict(data.features[row_ndx, :], weights):
            num_correct += 1

    return 1.0 - float(num_correct / data.features.shape[0])
