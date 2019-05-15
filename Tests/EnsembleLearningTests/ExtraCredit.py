from DecisionTree.Id3 import get_prediction
from Data.credit import CreditDefaultData
from EnsembleLearning import AdaBoost
from EnsembleLearning import BaggingTrees
from EnsembleLearning import RandomForests
from DecisionTree import Id3
from DecisionTree import Metrics
import matplotlib.pyplot as plt
import os
import sys
import numpy as np


def extra_credit():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = CreditDefaultData.Data()
    data.initialize_data_from_file(dir_path + '/../Data/credit/credit.csv')

    # t_values = [1, 2, 5, 10, 25, 50, 75, 125, 250, 500, 1000]
    iterates = [1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 100]
    t_sum = np.sum(t_values)

    ada_boost_train_error = []
    ada_boost_test_error = []
    bagging_trees_train_error = []
    bagging_trees_test_error = []
    random_forests_train_error = []
    random_forests_test_error = []

    ada_trees = AdaBoost.Adaboost(features=data.f, attributes=data.attributes, labelset=data.labels, t_value=100)
    bag_trees = BaggingTrees.run_bagging_trees(100, data.train_examples, data.attributes, data.labels, 2, True)
    r_forest = RandomForests.run_random_forests(100, data.train_examples, data.attributes, data.labels, 4, True)
    print("Tree construction complete. Calculating Data.")

    counter = int(100 / t_sum)
    toolbar_width = 100
    # sys.stdout.write("Progress: [%s]" % (" " * toolbar_width))
    # sys.stdout.flush()

    for num_iterates in iterates:
        ada_boost_train_error.append(get_ada_error(data.train_examples, data, ada_trees, num_iterates))
        ada_boost_test_error.append(get_ada_error(data.test_examples, data, ada_trees, num_iterates))

        bagging_trees_train_error.append(get_bag_error(data.train_examples, data, bag_trees, num_iterates))
        bagging_trees_test_error.append(get_bag_error(data.test_examples, data, bag_trees, num_iterates))

        random_forests_train_error.append(get_bag_error(data.train_examples, data, r_forest, num_iterates))
        random_forests_test_error.append(get_bag_error(data.test_examples, data, r_forest, num_iterates))

        # sys.stdout.write('\r')
        # sys.stdout.flush()
        # sys.stdout.write('Progress: [%s' % ('#' * counter))
        # sys.stdout.write('%s]' % (' ' * (toolbar_width - counter)))
        # sys.stdout.flush()
        # fractor = float(t_value * 100 / t_sum)  # May be broken
        # counter += int(fractor)

    print("")
    plt.plot(t_values, ada_boost_train_error, label='Ada Train')
    plt.plot(t_values, ada_boost_test_error, label='Ada Test')
    plt.plot(t_values, bagging_trees_train_error, label='Bag Train', marker='<')
    plt.plot(t_values, bagging_trees_test_error, label='Bag Test', marker='<')
    plt.plot(t_values, random_forests_train_error, label='Forest Train', marker='o')
    plt.plot(t_values, random_forests_test_error, label='Forest Test', marker='o')
    plt.legend(loc='best')
    plt.show()

    print("Now running statistics on a fully developed decision tree.")
    id3 = Id3.Id3(metric='information_gain')
    dec_tree = id3.run_id3(data.train_examples, data.attributes, None, data.labels, 0, float("inf"))
    dec_tree_train_error = get_dec_error(data.train_examples, data, dec_tree)
    dec_tree_test_error = get_dec_error(data.test_examples, data, dec_tree)
    print("Train error for full decision tree is %.16f" % dec_tree_train_error)
    print("Test error for full decision tree is %.16f" % dec_tree_test_error)


def get_ada_error(examples, data, trees, t_value):
    correct_results = 0
    for example in examples:
        if example.get_label() == AdaBoost.get_final_hypothesis(t_value, trees[1], example, trees[0], data):
            correct_results += 1

    return 1.0 - float(correct_results) / float(len(examples))


def get_bag_error(examples, data, trees, t_value):
    correct_results = 0
    for example in examples:
        if example.get_label() == BaggingTrees.get_result(example, trees, data, t_value):
            correct_results += 1

    return 1.0 - float(correct_results) / float(len(examples))


def get_dec_error(examples, data, root):
    correct_results = 0
    for example in examples:
        if example.get_label() == get_prediction(example, root):
            correct_results += 1

    return 1.0 - float(correct_results) / float(len(examples))
