import numpy as np
import random


def perceptron(num_epochs, features, labels, r_step):
    w_vector = np.zeros(features.shape[1])

    for i in range(0, num_epochs):
        shuffled_indices = shuffle(features.shape[0])
        for ndx in shuffled_indices:
            amt = labels[ndx] * (features[ndx, :]).dot(w_vector)
            if amt <= 0.0:
                w_vector = w_vector + r_step * labels[ndx] * features[ndx, :]

    return w_vector


def voted_perceptron(num_epochs, features, labels, r_step):
    w_vector = np.zeros(features.shape[1])
    ret = [[], []]
    m = 0
    c = 1

    for i in range(0, num_epochs):
        shuffled_indices = shuffle(features.shape[0])
        for ndx in shuffled_indices:
            amt = labels[ndx] * (features[ndx, :]).dot(w_vector)
            if amt <= 0.0:
                w_vector = w_vector + r_step * labels[ndx] * features[ndx, :]
                m += 1
                ret[1].append(c)
                ret[0].append(w_vector)
                c = 1

            else:
                c += 1

    return ret


def averaged_perceptron(num_epochs, features, labels, r_step):
    w_vector = np.zeros(features.shape[1])
    a_vector = np.zeros(features.shape[1])

    for i in range(0, num_epochs):
        shuffled_indices = shuffle(features.shape[0])
        for ndx in shuffled_indices:
            amt = labels[ndx] * (features[ndx, :]).dot(w_vector)
            if amt <= 0.0:
                w_vector = w_vector + r_step * labels[ndx] * features[ndx, :]
            else:
                a_vector = a_vector + w_vector

    return a_vector


def shuffle(size):
    return random.sample(range(size), size)


# Used for both standard and averaged perceptrons
def get_prediction(w_vector, x_vector):
    return np.sign(w_vector.dot(x_vector))


def get_voted_prediction(voted_ret_array, x_vector):
    _sum = 0.0
    for i in range(0, len(voted_ret_array[0])):
        _sum += voted_ret_array[1][i] * np.sign(voted_ret_array[0][i].dot(x_vector))
    return np.sign(_sum)


class Perceptron:
    """
    Perceptron class
    """
    def __init__(self, num_epochs, features, labels, r_step, perceptron_type='default'):
        """
        Perceptron constructor
        :param num_epochs: number of epochs for algorithm
        :type num_epochs: integer
        :param features: data features
        :type features: numpy array
        :param labels: corresponding data labels
        :type labels: numpy array
        :param r_step: learning rate:
        :type r_step: float
        :param perceptron_type: optional parameter. Enter 'averaged' or 'voted'. If neither, the default is used
        :type perceptron_type: string
        """
        self.perceptron_type = perceptron_type
        if perceptron_type not in ['default', 'averaged', 'voted']:
            raise ValueError('Incorrect perceptron type [%s] encountered in constructor' % self.perceptron_type)
        self.num_epochs = num_epochs
        self.features = features
        self.labels = labels
        self.r_step = r_step
        self.weights = None

    def fit(self):
        """
        fit perceptron algorithm
        :return:
        """
        if self.perceptron_type == 'default':
            self.weights = perceptron(num_epochs=self.num_epochs, features=self.features, labels=self.labels,
                                      r_step=self.r_step)
        elif self.perceptron_type == 'averaged':
            self.weights = averaged_perceptron(num_epochs=self.num_epochs, features=self.features, labels=self.labels,
                                               r_step=self.r_step)
        else:
            self.weights = voted_perceptron(num_epochs=self.num_epochs, features=self.features, labels=self.labels,
                                            r_step=self.r_step)

    def predict(self, x):
        """
        predict label for feature x
        :param x: data feature
        :type x: numpy array
        :return: +/- 1.0
        :rtype: float
        """
        if self.perceptron_type == 'voted':
            return get_voted_prediction(self.weights, x)
        else:
            return get_prediction(self.weights, x)
