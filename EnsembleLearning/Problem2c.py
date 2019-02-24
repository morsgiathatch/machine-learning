from DecisionTree import BankData
from EnsembleLearning import BaggingTrees
from DecisionTree import Id3
from DecisionTree import Metrics
import sys
import os
import math
import random


def problem2c():
    # Train data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankData.Data()
    data.initialize_data_from_file(dir_path + '/../Data/bank/train.csv', False)

    # Test data
    test_data = BankData.Data()
    test_data.initialize_data_from_file(dir_path + '/../Data/bank/test.csv', False)

    bagged_trees = []
    full_trees = []

    factor = float(input("Please enter a number to get fraction of examples. (e.g., `2` uses 1/2 of examples):\n"))

    for i in range(0, 100):
        # sample 100 examples uniformly without replacement
        print("Begin calculations for " + str(i) + "th run")
        examples = get_samples(data)
        bagged_trees.append(BaggingTrees.run_bagging_trees(1000, examples, data.attributes, data.labels, factor))
        print("Got bagged trees")
        id3 = Id3.Id3()
        full_trees.append(id3.id3(examples, data.attributes, None, data.labels, 0, float("inf"), Metrics.information_gain))
        print("Got full trees")

    print("Calculating squared mean error of full trees.")
    full_trees_mean_squared_error = get_squared_mean_error(data, full_trees, False)
    print("\nMean Squared Error for the full trees is: " + "%.16f" % full_trees_mean_squared_error)
    print("Calculating squared mean error for bagged trees.")
    bagged_trees_mean_squared_error = get_squared_mean_error(data, bagged_trees, True)
    print("Mean Squared Error for the bagged trees is: " + "%.16f" % bagged_trees_mean_squared_error)


def get_samples(data):
    examples = []
    indices = []

    while len(examples) != 1000:
        index = random.randint(0, len(data.examples) - 1)
        if index not in indices:
            indices.append(index)
            examples.append(data.examples[index])

    return examples


def get_squared_mean_error(data, trees, bagged_tree):
    bias = 0.0
    variance = 0.0
    toolbar_width = 100
    sys.stdout.write("Progress:")
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))

    counter = 0
    subdivision = int(len(data.examples) / 100)
    for example in data.examples:
        _sum = 0.0
        if bagged_tree:
            for tree_set in trees:
                _sum += BaggingTrees.get_result(example, tree_set, data)
        else:
            for tree in trees:
                _sum += data.get_test_result(example, tree)

        tree_avg = _sum / float(len(trees))

        bias += math.pow(example.get_label() - tree_avg, 2.0)

        _sum = 0.0
        if bagged_tree:
            for tree_set in trees:
                _sum += math.pow(BaggingTrees.get_result(example, tree_set, data) - tree_avg, 2.0)
        else:
            for tree in trees:
                _sum += math.pow(data.get_test_result(example, tree) - tree_avg, 2.0)
        variance += _sum / (float(len(trees)) - 1.0)

        counter += 1
        if counter == subdivision:
            counter = 0
            sys.stdout.write("#")
            sys.stdout.flush()

    sys.stdout.write("\n")
    bias /= float(len(data.examples))
    variance /= float(len(data.examples))
    return bias + variance

