import numpy as np
import random
from numpy import linalg as la


class KernelPerceptron:
    """
    KernelPerceptron class
    """
    def __init__(self, gamma, features, labels):
        """
        KernelPerceptron constructor
        :param gamma: hyper-parameter used in kernel
        :type gamma: float
        :param features: data features
        :type features: numpy array
        :param labels: corresponding data labels
        :type labels: numpy array
        """
        self.gamma = gamma
        self.features = features
        self.labels = labels
        self.kernel = self.get_kernel_matrix()
        self.counts = None

    def fit(self, num_epochs):
        """
        fit kernel perceptron algorithm
        :param num_epochs: number of epochs
        :type num_epochs: integer
        :return: c*y X
        :rtype: numpy array
        """
        counts = np.zeros(self.features.shape[0])
        for i in range(0, num_epochs):
            [shuffled_indices, counts] = shuffle(self.features.shape[0], counts)
            for ndx in shuffled_indices:
                amt = 0.0
                for j in range(0, self.features.shape[0]):
                    amt += counts[j] * self.labels[j] * self.kernel[ndx, j]
                if int(self.labels[ndx]) != np.sign(amt):
                    counts[ndx] += 1

        self.counts = counts
        return np.matmul(np.multiply(counts, self.labels), self.features)

    # Used for both standard and averaged perceptrons
    def predict(self, x):
        """
        predict label for feature
        :param x: feature
        :type x: numpy array
        :return: +/- 1.0
        :rtype: float
        """
        _sum = 0.0
        for i in range(0, self.features.shape[0]):
            _sum += self.counts[i] * self.labels[i] * self.gaussian_kernel(self.features[i, :], x)

        return np.sign(_sum)

    def gaussian_kernel(self, x, y):
        return np.exp(-1.0 * (la.norm(x - y) ** 2) / self.gamma)

    def get_kernel_matrix(self):
        kernel = np.zeros(shape=(self.features.shape[0], self.features.shape[0]))
        for i in range(0, self.features.shape[0]):
            for j in range(9, self.features.shape[0]):
                kernel[i, j] = self.gaussian_kernel(self.features[i, :], self.features[j, :])

        for i in range(0, self.features.shape[0]):
            for j in range(0, i):
                kernel[i, j] = kernel[j, i]

        return kernel


def shuffle(size, counts_orig):
    indices = random.sample(range(size), size)
    counts = np.zeros(size)
    for i, ndx in enumerate(indices):
        counts[i] = counts_orig[ndx]

    return [indices, counts]
