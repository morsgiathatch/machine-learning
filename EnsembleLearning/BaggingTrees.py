from DecisionTree import Id3
from DecisionTree import Metrics
import numpy as np
import random


def run_bagging_trees(t_value, data_examples, attributes, labels, factor):

    trees = []

    for i in range(0, t_value):
        examples = get_sample(data_examples, factor)

        # Run Id3 and keep root node
        id3 = Id3.Id3()
        trees.append(id3.id3(examples, attributes, None, labels, 0, float("inf"), Metrics.information_gain))

    return trees


def get_sample(examples, factor):
    samples = []
    for j in range(0, int(len(examples) / factor)):
        samples.append(examples[random.randint(0, len(examples) - 1)])
    return samples


def get_result(example, trees, data):
    _sum = 0.0
    for tree in trees:
        _sum += data.get_test_result(example, tree)

    _sum /= float(len(trees))

    return np.sign(_sum)
