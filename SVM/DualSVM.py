import numpy as np
from numpy import linalg as la


def svm(a, data, XXt):
    return 0.5 * np.matmul(np.multiply(a, data.output), np.matmul(XXt, np.multiply(a, data.output))) - np.sum(a)


def generate_bounds(a, C):
    bounds = []
    for i in range(0, a.shape[0]):
        bounds.append((0.0, C))

    return tuple(bounds)


def grad_svm(a, data, XXt):
    return np.multiply(np.matmul(np.multiply(a, data.output), XXt), data.output) - np.ones(a.shape[0])


def a_dot_y(a, data):
    return a.dot(data.output)


def gaussian_kernel(x, y, gamma):
    return np.exp(-1.0 * (la.norm(x - y) ** 2) / gamma)


def get_kernel_prediction(alphas, x, gamma, data):
    _sum = 0.0
    for i in range(0, alphas.shape[0]):
        _sum += data.output[i] * alphas[i] * gaussian_kernel(data.features[i, :], x, gamma)
    return np.sign(_sum)


def get_percentages(test_data, data, alphas, gamma):
    num_correct = 0
    for row_ndx in range(0, test_data.features.shape[0]):
        if test_data.output[row_ndx] == get_kernel_prediction(alphas, test_data.features[row_ndx, :], gamma, data):
            num_correct += 1

    return 1.0 - float(num_correct / test_data.features.shape[0])


def get_XXt(use_kernel_prediction, gamma, data):
    if use_kernel_prediction:
        XXt = np.zeros(shape=(data.features.shape[0], data.features.shape[0]))
        for i in range(0, data.features.shape[0]):
            for j in range(i, data.features.shape[0]):
                XXt[i, j] = gaussian_kernel(data.features[i, :], data.features[j, :], gamma)

        for i in range(0, data.features.shape[0]):
            for j in range(0, i):
                XXt[i, j] = XXt[j, i]
    else:
        XXt = np.matmul(data.features, data.features.transpose())

    return XXt
