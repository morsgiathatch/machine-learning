from numpy import linalg as la
import numpy as np


# Functions and gradient functions to be passed to gradient descent
def objective_function(features, labels, args):
    """
    objective function to pass to Algorithms/GradientDescent object fit/fit_stochastic function

    :param features: data features
    :type features: numpy array
    :param labels: data labels
    :type labels: numpy array
    :param args: weights for gradient descent
    :type args: python list of numpy array
    :return: 1/2 ||X * w - y||
    :rtype: float
    """
    w_vector = args[0]
    return 0.5 * la.norm(np.matmul(features, w_vector) - labels)


def obj_gradient_function(features, labels, args):
    """
    gradient function to pass to Algorithms/GradientDescent object fit function

    :param features: data features
    :type features: numpy array
    :param labels: data labels
    :type labels: numpy array
    :param args: weights for gradient descent
    :type args: python list of numpy array
    :return: X^t(X * w - y)
    :rtype: numpy array
    """
    weights = args[0]
    return (features.transpose()).dot((features.dot(weights)) - labels)


def stoch_gradient_function(feature, label, args):
    """
    gradient function to pass to Algorithms/GradientDescent object fit_stochastic function

    :param feature: data features
    :type feature: numpy array
    :param label: data labels
    :type label: numpy array
    :param args: weights for gradient descent
    :type args: python list of numpy array
    :return: (y_i - w * x_i)x_i
    :rtype: float
    """
    weights = args[0]
    return (weights.dot(feature) - label) * feature


def analytic_solution(features, labels):
    """
    analytic solution to objective function 1/2 ||X * w|| - y

    :param features: data features
    :type features: numpy array
    :param labels: data labels
    :type labels: numpy array
    :return: (X^t * X)^(-1)(X^t * y)
    :rtype: numpy array
    """
    return (la.inv((features.transpose()).dot(features))).dot((features.transpose()).dot(labels))
