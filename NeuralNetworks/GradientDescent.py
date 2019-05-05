import random
import sys
import numpy as np


class GradientDescent:
    def __init__(self, features, labels, gamma_0, d):
        self.features = features
        self.labels = labels
        self.gamma_0 = gamma_0
        self.d = d

    def run_stochastic_sub_grad_descent(self, max_iters, obj_func, grad_func, weights, args):
        # initialize weight vector
        objective_function_values = []

        for t in range(0, max_iters + 1):

            if t == max_iters:
                break

            shuffled_indices = random.sample(range(0, self.features.shape[0]), self.features.shape[0])

            for i, index in enumerate(shuffled_indices):
                gamma_t = self.gamma_0 / (1.0 + (self.gamma_0 / self.d) * t)
                weights -= gamma_t * grad_func(self.features[index, :], self.labels[index], [weights, args])
            sys.stdout.write("\r%i / %i" % (t + 1, max_iters))
            sys.stdout.flush()
            objective_function_values.append(obj_func(self.features, self.labels, [weights, args]))

        sys.stdout.write('\n')
        sys.stdout.flush()
        return [weights, np.array(objective_function_values)]
        # return weights
