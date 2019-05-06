from DecisionTree import Id3
from DecisionTree import Metrics
import numpy as np
import random
import sys

from DecisionTree.Id3 import get_prediction


def run_bagging_trees(t_value, data_examples, attributes, labels, factor, print_status_bar):

    trees = []

    counter = 1
    fractor = int(t_value / 100)
    toolbar_width = 100
    if print_status_bar:
        print("Building Bagging Trees")
        sys.stdout.write("Progress: [%s]" % (" " * toolbar_width))
        sys.stdout.flush()
    for i in range(0, t_value):
        examples = get_sample(data_examples, factor)

        # Run Id3 and keep root node
        id3 = Id3.Id3()
        trees.append(id3.id3(examples, attributes, None, labels, 0, float("inf"), Metrics.information_gain))

        if i % fractor == 0 and print_status_bar:
            sys.stdout.write('\r')
            sys.stdout.flush()
            sys.stdout.write('Progress: [%s' % ('#' * counter))
            sys.stdout.write('%s]' % (' ' * (toolbar_width - counter)))
            sys.stdout.flush()
            counter += 1

    if print_status_bar:
        print("")

    return trees


def get_sample(examples, factor):
    samples = []
    for j in range(0, int(len(examples) / factor)):
        samples.append(examples[random.randint(0, len(examples) - 1)])
    return samples


def get_result(example, trees, data, t_value):
    _sum = 0.0
    for i in range(0, t_value):
        _sum += get_prediction(example, trees[i])

    _sum /= float(len(trees))

    return np.sign(_sum)
