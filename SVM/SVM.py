import numpy as np


def stoch_grad_func(feature, label, args):
    weights = args[0]
    C = args[1][0]
    N = args[1][1]
    weights_temp = np.append(weights[:-1], 0.0)
    if np.maximum(0.0, 1.0 - label * weights.dot(feature)) != 0.0:
        weights_temp -= C * N * label * feature
    return weights_temp


def objective_function(features, labels, args):
    weights = args[0]
    C = args[1][0]
    yXw = -np.multiply(np.matmul(features, weights), labels)
    yXw += np.ones(yXw.shape[-1])
    return 0.5 * weights[:-1].dot(weights[:-1]) + C * np.sum(np.maximum(np.zeros(yXw.shape[-1]), yXw))
