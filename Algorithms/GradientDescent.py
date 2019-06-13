import numpy as np
from numpy import linalg as la
import random
import sys


class GradientDescent:
    """
    GradientDescent class
    """
    def __init__(self, features, labels):
        """
        GradientDescent constructor

        :param features: data features
        :type features: numpy array
        :param labels: binary labels for each feature
        :type labels: numpy array
        """
        self.features = features
        self.labels = labels

    # Runs gradient descent. max_iters is the number of desired epochs, obj_func, grad_func are function references
    # to calculate the objective function and gradient function. args is a python list of arguments for both
    # obj_func and grad_func. step_function is a function reference to calculate the step the next step function. It
    # must only take one argument for the current iteration. weights is an optional argument if the user wishes to
    # pass a non-zero initial weight. tolerance is an optional argument if the user wishes to have a terminating
    # condition on the gradient descent. norm is an optional argument for the terminating condition, defaults to the
    # euclidean norm. print_status is an optional argument that prints a status for the user, default is True.
    def fit(self, max_iters, obj_func, grad_func, args, step_function, weights=None,
            tolerance=None, norm=la.norm, print_status=True):
        """
        fit gradient descent to data

        :param max_iters: number of epochs
        :type max_iters: integer
        :param obj_func: function reference for the objective function to be used
        :type obj_func: function
        :param grad_func: gradient of objective function
        :type grad_func: function
        :param args: agruments for both obj_func and grad_func
        :type args: python list
        :param step_function: function reference to calculate next step size or learning rate
        :type step_function: function with one parameter taking an integer representing epoch number
        :param weights: optional argument to use pre-determined weights. zero array by default
        :type weights: numpy array
        :param tolerance: optional stopping condition
        :type tolerance: float
        :param norm: optional function reference to the norm used to find iterate error, defaults to euclidean norm
        :type  norm: function
        :param print_status: optional argument to print out status updates
        :type print_status: boolean
        :return: returns [weights, terminating iteration, objective function values] if tolerance set. Otherwise
        [weights, objective function values]
        :rtype: python list
        """
        # initialize weight vector if not given initial weight vector
        if weights is None:
            weights = np.zeros(self.features.shape[1])

        # Initialize storage for objective function values to analyze convergence
        objective_function_values = []
        num_iters = 0

        for t in range(0, max_iters + 1):
            num_iters = t + 1
            if t == max_iters:
                break

            objective_function_values.append(obj_func(self.features, self.labels, [weights, args]))

            old_weights = np.copy(weights)
            weights -= step_function(t) * grad_func(self.features, self.labels, [weights, args])

            if tolerance is not None and num_iters != 1:
                if norm(weights - old_weights) <= tolerance:
                    break

            if print_status:
                sys.stdout.write("\r%i / %i" % (t + 1, max_iters))
                sys.stdout.flush()
            objective_function_values.append(obj_func(self.features, self.labels, [weights, args]))
            num_iters += 1

        if print_status:
            sys.stdout.write('\n')
            sys.stdout.flush()
        # return weights
        if tolerance is not None:
            return [weights, num_iters, np.array(objective_function_values)]

        return [weights, np.array(objective_function_values)]

    # Runs stochastic gradient descent. max_iters is the number of desired epochs, obj_func, grad_func are function
    # references to calculate the objective function and gradient function. args is a python list of arguments for both
    # obj_func and grad_func. step_function is a function reference to calculate the step the next step function. It
    # must only take one argument for the current iteration. weights is an optional argument if the user wishes to
    # pass a non-zero initial weight. tolerance is an optional argument if the user wishes to have a terminating
    # condition on the gradient descent. norm is an optional argument for the terminating condition, defaults to the
    # euclidean norm. print_status is an optional argument that prints a status for the user, default is True.
    def fit_stochastic(self, max_iters, obj_func, grad_func, args, step_function,
                       weights=None, tolerance=None, norm=la.norm, print_status=True):
        """
        fit stochastic gradient descent

        :param max_iters: number of epochs
        :type max_iters: integer
        :param obj_func: function reference for the objective function to be used
        :type obj_func: function
        :param grad_func: gradient of objective function
        :type grad_func: function
        :param args: agruments for both obj_func and grad_func
        :type args: python list
        :param step_function: function reference to calculate next step size or learning rate
        :type step_function: function with one parameter taking an integer representing epoch number
        :param weights: optional argument to use pre-determined weights. zero array by default
        :type weights: numpy array
        :param tolerance: optional stopping condition
        :type tolerance: float
        :param norm: optional function reference to the norm used to find iterate error, defaults to euclidean norm
        :type  norm: function
        :param print_status: optional argument to print out status updates
        :type print_status: boolean
        :return: returns [weights, terminating iteration, objective function values] if tolerance set. Otherwise
        [weights, objective function values]
        :rtype: python list
        """
        # initialize weight vector if not given initial weight vector
        if weights is None:
            weights = np.zeros(self.features.shape[1])

        # Initialize storage for objective function values to analyze convergence
        objective_function_values = []
        num_iters = 0

        for t in range(0, max_iters + 1):
            if t == max_iters:
                break

            shuffled_indices = random.sample(range(0, self.features.shape[0]), self.features.shape[0])
            objective_function_values.append(obj_func(self.features, self.labels, [weights, args]))

            if t > 280:
                a = 3
            old_weights = np.copy(weights)
            for i, index in enumerate(shuffled_indices):
                weights -= step_function(t) * grad_func(self.features[index, :], self.labels[index], [weights, args])

            if tolerance is not None and t > 0:
                if norm(weights - old_weights) <= tolerance:
                    break

            if print_status:
                sys.stdout.write("\r%i / %i" % (t + 1, max_iters))
                sys.stdout.flush()
            num_iters += 1

        if print_status:
            sys.stdout.write('\n')
            sys.stdout.flush()

        # return weights
        if tolerance is not None:
            return [weights, num_iters, np.array(objective_function_values)]

        return [weights, np.array(objective_function_values)]
