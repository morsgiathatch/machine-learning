import numpy as np
import random


def perceptron(num_epochs, examples, labels, r_step):
    w_vector = np.zeros(examples.shape[1])

    for i in range(0, num_epochs):
        shuffled_indices = shuffle(examples.shape[0])
        for ndx in shuffled_indices:
            amt = labels[ndx] * (examples[ndx, :]).dot(w_vector)
            if amt <= 0.0:
                w_vector = w_vector + r_step * labels[ndx] * examples[ndx, :]

    return w_vector


def voted_perceptron(num_epochs, examples, labels, r_step):
    w_vector = np.zeros(examples.shape[1])
    ret = [[], []]
    m = 0
    c = 1

    for i in range(0, num_epochs):
        shuffled_indices = shuffle(examples.shape[0])
        for ndx in shuffled_indices:
            amt = labels[ndx] * (examples[ndx, :]).dot(w_vector)
            if amt <= 0.0:
                w_vector = w_vector + r_step * labels[ndx] * examples[ndx, :]
                m += 1
                ret[1].append(c)
                ret[0].append(w_vector)
                c = 1

            else:
                c += 1

    return ret


def averaged_perceptron(num_epochs, examples, labels, r_step):
    w_vector = np.zeros(examples.shape[1])
    a_vector = np.zeros(examples.shape[1])
    ret = [[], []]

    for i in range(0, num_epochs):
        shuffled_indices = shuffle(examples.shape[0])
        for ndx in shuffled_indices:
            amt = labels[ndx] * (examples[ndx, :]).dot(w_vector)
            if amt <= 0.0:
                w_vector = w_vector + r_step * labels[ndx] * examples[ndx, :]
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
