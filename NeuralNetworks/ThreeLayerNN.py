import numpy as np


class ThreeLayerNN:
    """
    ThreeLayerNN class
    """
    def __init__(self, num_units_per_layer, weights):   # if we have w_{ij}^k, call weights[i, j, k] to get it
        """
        ThreeLayerNN constructor

        :param num_units_per_layer: number of units per layer. Each layer will have the same number of units
        :type num_units_per_layer: integer
        :param weights: desired initial weights
        :type weights: numpy array of shape (, , )
        """
        self.num_units_per_layer = num_units_per_layer
        self.weights = weights
        self.layer0 = None
        self.layer1 = np.array([1.0] * self.num_units_per_layer)
        self.layer2 = np.array([1.0] * self.num_units_per_layer)

    # Call this to predict a label for example x
    def predict(self, x):
        """
        Predict label of feature x after model has been fitted

        :param x: feature
        :type x: numpy array
        :return: prediction
        :rtype: float
        """
        self.layer0 = np.copy(x)

        for i in range(1, self.num_units_per_layer):
            self.layer1[i] = sigmoid(x.dot(self.weights[0:x.shape[0], i, 1]))

        for i in range(1, self.num_units_per_layer):
            self.layer2[i] = sigmoid(self.layer1.dot(self.weights[:, i, 2]))

        return np.sign(self.layer2.dot(self.weights[0:self.layer2.shape[0], 1, 3]))

    def update_weights(self, weights):
        self.weights = weights

    def gradient(self, x, y_actual, args):
        """
        gradient function using backtracking to be passed to gradient descent

        :param x: training features
        :type x: numpy array
        :param y_actual: training labels
        :type y_actual: numpy array
        :param args: list containing weights [weights], weights is a numpy array
        :type args: python list
        :return: weights
        :rtype: numpy array
        """
        weights = args[0]
        self.update_weights(weights)
        # Update zeroth layer
        self.layer0 = x.tolist()

        # Begin backtracking
        y = self.predict(x)
        grad_cache = np.zeros((self.num_units_per_layer, self.num_units_per_layer, 4))
        grad_cache.fill(0.0)

        # Find 3rd layer of derivatives
        for i in range(0, self.num_units_per_layer):
            grad_cache[i, 1, 3] = (y - y_actual) * self.layer2[i]

        # Find 2nd layer of derivatives
        for i in range(0, self.num_units_per_layer):
            for j in range(1, self.num_units_per_layer):
                grad_cache[i, j, 2] = grad_cache[j, 1, 3] * self.weights[j, 1, 3] * (1.0 - self.layer2[j]) * self.layer1[i]

        # Find 3rd layer of derivatives
        for i in range(0, x.shape[0]):
            for j in range(1, self.num_units_per_layer):
                grad_cache[i, j, 1] = x[i] * (1.0 - self.layer1[j]) * np.sum(np.multiply(self.weights[j, :, 2], grad_cache[j, :, 2]))

        return grad_cache

    def objective_function(self, features, labels, extra_args=None):
        """
        Objective function to be passed to gradient descent fit_stochastic

        :param features: data features
        :type features: numpy array
        :param labels: corresponding data labels
        :type labels: numpy array
        :param extra_args: Useless argument to fit API
        :type extra_args: None
        :return: objective function value
        :rtype: float
        """
        _sum = 0.0
        for i in range(0, features.shape[0]):
            _sum += (self.predict(features[i, :]) - labels[i]) ** 2
        return 0.5 * _sum


def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-1.0 * val))
