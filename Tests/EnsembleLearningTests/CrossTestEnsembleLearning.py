from Data.credit import CreditDefaultData
from EnsembleLearning import AdaBoost
from EnsembleLearning import BaggingTrees
from EnsembleLearning import RandomForests
from DecisionTree import Id3
import matplotlib.pyplot as plt
import os
import sys
import numpy as np


# Run cross comparison on different test set
def run_cross_comparison():
    print_error_calculation_status = False
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = CreditDefaultData.Data()
    data.initialize_data_from_file(dir_path + '/../../Data/credit/credit.csv')

    iterates = [1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 100]
    it_sum = float(np.sum(iterates))

    ada_boost_train_error = []
    ada_boost_test_error = []
    bagging_trees_train_error = []
    bagging_trees_test_error = []
    random_forests_train_error = []
    random_forests_test_error = []

    ada_trees = AdaBoost.Adaboost(features=data.train_examples, attributes=data.attributes, t_value=100)
    ada_trees.fit(print_status=True)
    bag_trees = BaggingTrees.BaggingTrees(features=data.train_examples, attributes=data.attributes, t_value=100,
                                          attribute_factor=2)
    bag_trees.fit(print_status_bar=True)
    r_forest = RandomForests.RandomForests(features=data.train_examples, attributes=data.attributes, t_value=100,
                                           size=4)
    r_forest.fit(print_status_bar=True)
    print("Tree construction complete. Calculating Data.")

    toolbar_width = 100
    if print_error_calculation_status:
        sys.stdout.write("Progress: [%s]" % (" " * toolbar_width))
        sys.stdout.flush()

    for i in range(0, len(iterates)):
        ada_boost_train_error.append(get_error(data.train_examples, ada_trees))
        ada_boost_test_error.append(get_error(data.test_examples, ada_trees))

        bagging_trees_train_error.append(get_error(data.train_examples, bag_trees))
        bagging_trees_test_error.append(get_error(data.test_examples, bag_trees))

        random_forests_train_error.append(get_error(data.train_examples, r_forest))
        random_forests_test_error.append(get_error(data.test_examples, r_forest))

        if print_error_calculation_status:
            counter = int((float(np.sum(iterates[:i + 1])) / it_sum) * toolbar_width)
            sys.stdout.write('\r')
            sys.stdout.flush()
            sys.stdout.write('Progress: [%s' % ('#' * counter))
            sys.stdout.write('%s]' % (' ' * (toolbar_width - counter)))
            sys.stdout.flush()

    print("")
    plt.plot(iterates, ada_boost_train_error, label='Ada Train')
    plt.plot(iterates, ada_boost_test_error, label='Ada Test')
    plt.plot(iterates, bagging_trees_train_error, label='Bag Train', marker='<')
    plt.plot(iterates, bagging_trees_test_error, label='Bag Test', marker='<')
    plt.plot(iterates, random_forests_train_error, label='Forest Train', marker='o')
    plt.plot(iterates, random_forests_test_error, label='Forest Test', marker='o')
    plt.legend(loc='best')
    plt.show()

    print("Now running statistics on a fully developed decision tree.")
    id3 = Id3.Id3(metric='information_gain')
    id3.fit(data.train_examples, data.attributes, None, data.labels, 0, float("inf"))
    dec_tree_train_error = get_error(data.train_examples, id3)
    dec_tree_test_error = get_error(data.test_examples, id3)
    print("Train error for full decision tree is %.16f" % dec_tree_train_error)
    print("Test error for full decision tree is %.16f" % dec_tree_test_error)


def get_error(examples, trees):
    correct_results = 0
    for example in examples:
        if example.get_label() == trees.predict(example):
            correct_results += 1

    return 1.0 - float(correct_results) / float(len(examples))
