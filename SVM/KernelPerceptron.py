import numpy as np
import random
from numpy import linalg as la


class KernelPerceptron:

    def __init__(self, gamma, examples, labels):
        self.gamma = gamma
        self.examples = examples
        self.labels = labels
        self.kernel = self.get_kernel_matrix()
        self.counts = None

    def run_kernel_perceptron(self, num_epochs):
        counts = np.zeros(self.examples.shape[0])
        for i in range(0, num_epochs):
            [shuffled_indices, counts] = shuffle(self.examples.shape[0], counts)
            for ndx in shuffled_indices:
                amt = 0.0
                for j in range(0, self.examples.shape[0]):
                    amt += counts[j] * self.labels[j] * self.kernel[ndx, j]
                if int(self.labels[ndx]) != np.sign(amt):
                    counts[ndx] += 1

        self.counts = counts
        return np.matmul(np.multiply(counts, self.labels), self.examples)

    # Used for both standard and averaged perceptrons
    def get_prediction(self, x):
        _sum = 0.0
        for i in range(0, self.examples.shape[0]):
            _sum += self.counts[i] * self.labels[i] * self.gaussian_kernel(self.examples[i, :], x)

        return np.sign(_sum)

    def gaussian_kernel(self, x, y):
        return np.exp(-1.0 * (la.norm(x - y) ** 2) / self.gamma)

    def get_kernel_matrix(self):
        kernel = np.zeros(shape=(self.examples.shape[0], self.examples.shape[0]))
        for i in range(0, self.examples.shape[0]):
            for j in range(9, self.examples.shape[0]):
                kernel[i, j] = self.gaussian_kernel(self.examples[i, :], self.examples[j, :])

        for i in range(0, self.examples.shape[0]):
            for j in range(0, i):
                kernel[i, j] = kernel[j, i]

        return kernel


def shuffle(size, counts_orig):
    indices = random.sample(range(size), size)
    counts = np.zeros(size)
    for i, ndx in enumerate(indices):
        counts[i] = counts_orig[ndx]

    return [indices, counts]
