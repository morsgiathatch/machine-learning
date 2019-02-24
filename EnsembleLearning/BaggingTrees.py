from DecisionTree import Id3
from DecisionTree import Metrics
import numpy as np
import random


def run_bagging_trees(t_value, data_examples, attributes, labels, factor):

    trees = []

    for i in range(0, t_value):

        # Construct subset of examples by generating random indices
        example_indices = []
        for j in range(0, int(len(data_examples) / factor)):
            example_indices.append(random.randint(0, len(data_examples) - 1))

        examples = []
        for j in example_indices:
            examples.append(data_examples[j])

        # Run Id3 and keep root node
        id3 = Id3.Id3()
        trees.append(id3.id3(examples, attributes, None, labels, 0, float("inf"), Metrics.information_gain))

    return trees


def get_result(example, trees, data):
    _sum = 0.0
    for tree in trees:
        _sum += data.get_test_result(example, tree)

    _sum /= len(trees)

    return np.sign(_sum)
