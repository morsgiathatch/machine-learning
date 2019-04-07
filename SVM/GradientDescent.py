import numpy as np
from numpy import linalg as la
import random


def run_stochastic_sub_grad_descent(features, output, max_iters, C, gamma_0, d, part):
    # initialize weight vector
    w_0_vector = np.zeros(features.shape[1] - 1)
    objective_function_values = []

    for t in range(0, max_iters + 1):

        if t == max_iters:
            break

        shuffled_indices = random.sample(range(0, features.shape[0]), features.shape[0])

        for index in shuffled_indices:

            if part == 'a':
                gamma_t = gamma_0 / (1 + (gamma_0 / d) * t)
            else:
                gamma_t = gamma_0 / (1 + t)

            temp_w = (w_0_vector.tolist())
            temp_w.append(0.0)
            w_vector = np.array(temp_w)

            if output[index] * w_vector.dot(features[index, :]) <= 1.0:

                w_vector = (1.0 - gamma_t) * w_vector + gamma_t * C * (features.shape[0]) * output[index] * features[index, :]
                temp_w = w_vector.tolist()
                temp_w.pop()
                w_0_vector = np.array(temp_w)

            else:
                w_0_vector = (1.0 - gamma_t) * w_0_vector

        objective_function_values.append(calculate_obj_func_val(w_vector, C, features, output))

    return [w_vector, np.array(objective_function_values)]


def get_analytic_solution(features, output):
    return (la.inv((features.transpose()).dot(features))).dot((features.transpose()).dot(output))


def get_cost(features, output, w_vector):
    return 0.5 * la.norm(features.dot(w_vector) - output)


def calculate_obj_func_val(w, C, X, Y):
    _sum = 0.0
    for i in range(0, X.shape[0]):
        _sum += np.maximum(0.0, 1.0 - Y[i] * w.dot(X[i, :]))

    return 0.5 * w.dot(w) + C * _sum