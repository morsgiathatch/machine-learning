import numpy as np


class ThreeLayerNN:
    def __init__(self, num_units_per_layer, weights):   # if we have w_{ij}^k, call weights[i, j, k] to get it
        self.num_units_per_layer = num_units_per_layer
        self.weights = weights
        self.layer0 = None
        self.layer1 = np.array([1.0] * self.num_units_per_layer)
        self.layer2 = np.array([1.0] * self.num_units_per_layer)

    # Call this to predict a label for example x
    def predict(self, x):
        self.layer0 = np.copy(x)

        for i in range(1, self.num_units_per_layer):
            self.layer1[i] = sigmoid(x.dot(self.weights[0:x.shape[0], i, 1]))

        for i in range(1, self.num_units_per_layer):
            self.layer2[i] = sigmoid(self.layer1.dot(self.weights[:, i, 2]))

        return np.sign(self.layer2.dot(self.weights[0:self.layer2.shape[0], 1, 3]))
        # return self.layer2.dot(self.weights[0:self.layer2.shape[0], 1, 3])

    def update_weights(self, weights):
        self.weights = weights

    def gradient(self, x, y_actual, args):
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
        _sum = 0.0
        for i in range(0, features.shape[0]):
            _sum += (self.predict(features[i, :]) - labels[i]) ** 2
        return 0.5 * _sum


def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-1.0 * val))
