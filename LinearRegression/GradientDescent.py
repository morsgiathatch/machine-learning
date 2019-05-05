import numpy as np
from numpy import linalg as la
import random


# Implementation of gradient descent uses constant step size. Input maximum termination
# condition in case convergence does not occur. Compact implementation is due thanks
# to a previous course in optimization where we covered this algorithm in depth
# mathematically. See Introduction to Nonlinear Optimization by Amir Beck, published by
# Society for Industrial and Applied Mathematics. This gradient descent algorithm is
# hard-coded to run on linear least squares only.

def run_gradient_descent(features, output, max_iters, constant_step_size, tolerance):
    # initialize weight vector
    w_vector = np.zeros(features.shape[1])

    evaluated_costs = []
    num_iters = 0
    for t in range(0, max_iters + 1):
        num_iters = t + 1
        if t == max_iters:
            break

        if t % 10 == 0:
            evaluated_costs.append(get_cost(features, output, w_vector))

        # w_(t + 1) = w_t - r* X^t (X * w_t - y)
        w_old = w_vector
        w_vector = w_vector - constant_step_size * (features.transpose()).dot((features.dot(w_vector)) - output)

        # || w_(t + 1) - w_t || <= epsilon ?
        if la.norm(w_vector - w_old) <= tolerance:
            break

    return [w_vector, num_iters, evaluated_costs]


def run_stochastic_grad_descent(features, output, max_iters, constant_step_size, tolerance):
    # initialize weight vector
    w_vector = np.zeros(features.shape[1])

    evaluated_costs = []
    num_iters = 0
    for t in range(0, max_iters + 1):
        rand_ndx = random.randint(0, features.shape[0] - 1)
        num_iters = t + 1
        if t == max_iters:
            break

        if t % 10 == 0:
            evaluated_costs.append(get_cost(features, output, w_vector))

        # w_(t + 1) = w_t - r* (y_i - x_i * w_t)x_i
        w_old = w_vector
        w_vector = w_vector + constant_step_size * \
                   (output[rand_ndx] - w_vector.dot(features[rand_ndx, :])) *(features[rand_ndx, :])

        # || w_(t + 1) - w_t || <= epsilon ?
        if la.norm(w_vector - w_old) <= tolerance:
            print("Success! Converged after " + str(num_iters) + " iterations.")
            break

    return [w_vector, num_iters, evaluated_costs]


def get_analytic_solution(features, output):
    return (la.inv((features.transpose()).dot(features))).dot((features.transpose()).dot(output))


def get_cost(features, output, w_vector):
    return 0.5 * la.norm(features.dot(w_vector) - output)
