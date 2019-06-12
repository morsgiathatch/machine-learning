import numpy as np


# functions to be passed to Algorithms/GradientDescent
def stoch_grad_func(feature, label, args):
    """
    stochastic gradient of primal svm objective function

    :param feature: data feature
    :type feature: numpy array
    :param label: corresponding data label
    :type label: float
    :param args: list in the form [weights] where weights is a numpy array
    :type args: python list
    :return: weights
    :rtype: numpy array
    """
    weights = args[0]
    C = args[1][0]
    N = args[1][1]
    weights_temp = np.append(weights[:-1], 0.0)
    if np.maximum(0.0, 1.0 - label * weights.dot(feature)) != 0.0:
        weights_temp -= C * N * label * feature
    return weights_temp


def objective_function(features, labels, args):
    """
    primal svm objective function

    :param features: data features
    :type features: numpy array
    :param labels: corresponding data labels
    :type labels: numpy array
    :param args: list in the form[weights] where weights is a numpy array
    :type args: python list
    :return: objective function value
    :rtype: float
    """
    weights = args[0]
    C = args[1][0]
    yXw = -np.multiply(np.matmul(features, weights), labels)
    yXw += np.ones(yXw.shape[-1])
    return 0.5 * weights[:-1].dot(weights[:-1]) + C * np.sum(np.maximum(np.zeros(yXw.shape[-1]), yXw))


def get_percentages(features, labels, weights):
    num_correct = 0
    for row_ndx in range(0, features.shape[0]):
        if labels[row_ndx] == np.sign(weights.dot(features[row_ndx, :])):
            num_correct += 1

    return 1.0 - float(num_correct / features.shape[0])
