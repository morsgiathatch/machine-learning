from DecisionTree import Id3
from DecisionTree import Metrics
import random
import numpy as np
import sys


def run_random_forests(t_value, examples, attributes, labels, size, print_status_bar):

    forest = []
    counter = 1
    factor = int(t_value / 100)
    toolbar_width = 100
    if print_status_bar:
        print("Building Bagging Trees")
        sys.stdout.write("Progress: [%s]" % (" " * toolbar_width))
        sys.stdout.flush()

    for i in range(0, t_value):
        # Get bootstrap example
        bootstrap_sample = get_bootstrap_sample(examples)
        id3 = Id3.RandomId3()
        forest.append(id3.random_id3(bootstrap_sample, attributes, None, labels, 0,
                              float("inf"), Metrics.information_gain, size))

        if i % factor == 0 and print_status_bar:
            sys.stdout.write('\r')
            sys.stdout.flush()
            sys.stdout.write('Progress: [%s' % ('#' * counter))
            sys.stdout.write('%s]' % (' ' * (toolbar_width - counter)))
            sys.stdout.flush()
            counter += 1

    if print_status_bar:
        print("")

    return forest


def get_bootstrap_sample(examples):
    bootstrap_samples = []
    for j in range(0, int(len(examples))):
        bootstrap_samples.append(examples[random.randint(0, len(examples) - 1)])
    return bootstrap_samples


def get_result(example, trees, data):
    _sum = 0.0
    for tree in trees:
        _sum += data.get_test_result(example, tree)

    _sum /= float(len(trees))

    return np.sign(_sum)
