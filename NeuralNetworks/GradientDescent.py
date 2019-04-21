import numpy as np
import random


class GradientDescent:
    def __init__(self, features, labels, gamma_0, d):
        self.features = features
        self.labels = labels
        self.gamma_0 = gamma_0
        self.d = d

    def run_stochastic_sub_grad_descent(self, max_iters, obj_func, grad_func):
        # initialize weight vector
        w_vector = np.zeros(self.features.shape[1])
        objective_function_values = []

        for t in range(0, max_iters + 1):

            if t == max_iters:
                break

            shuffled_indices = random.sample(range(0, self.features.shape[0]), self.features.shape[0])

            for index in shuffled_indices:

                gamma_t = self.gamma_0 / (1 + (self.gamma_0 / self.d) * t)

                w_vector = w_vector - gamma_t * grad_func(self.features[index, :], self.labels[index], w_vector)

            objective_function_values.append(obj_func(self.features, self.labels, w_vector))

        return [w_vector, np.array(objective_function_values)]
