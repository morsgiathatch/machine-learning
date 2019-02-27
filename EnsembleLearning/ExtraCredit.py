from EnsembleLearning import CreditDefaultData
from EnsembleLearning import AdaBoost
from EnsembleLearning import BaggingTrees
from EnsembleLearning import RandomForests
import matplotlib.pyplot as plt
import os
import sys


def extra_credit():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = CreditDefaultData.Data()
    data.initialize_data_from_file(dir_path + '/../Data/credit/credit.csv')

    t_values = [1, 2, 5, 10, 25, 50, 75, 125, 250, 500, 1000]
    # t_values = [1, 2, 5, 10]

    ada_boost_train_error = []
    ada_boost_test_error = []
    bagging_trees_train_error = []
    bagging_trees_test_error = []
    random_forests_train_error = []
    random_forests_test_error = []

    # counter = 1
    # toolbar_width = 100
    # print("Building trees")
    # sys.stdout.write("Progress: [%s]" % (" " * toolbar_width))
    # sys.stdout.flush()
    ada_trees = AdaBoost.run_credit_Adaboost(data, 1000)
    bag_trees = BaggingTrees.run_bagging_trees(1000, data.train_examples, data.attributes, data.labels, 2, True)
    r_forest = RandomForests.run_random_forests(1000, data.train_examples, data.attributes, data.labels, 4, True)
    print("Tree construction complete. Calculating Data.")

    for t_value in t_values:
        ada_boost_train_error.append(get_ada_error(data.train_examples, data, ada_trees, t_value))
        ada_boost_test_error.append(get_ada_error(data.test_examples, data, ada_trees, t_value))

        bagging_trees_train_error.append(get_bag_error(data.train_examples, data, bag_trees, t_value))
        bagging_trees_test_error.append(get_bag_error(data.test_examples, data, bag_trees, t_value))

        random_forests_train_error.append(get_bag_error(data.train_examples, data, r_forest))
        random_forests_test_error.append(get_bag_error(data.test_examples, data, r_forest))

        # sys.stdout.write('\r')
        # sys.stdout.flush()
        # sys.stdout.write('Progress: [%s' % ('#' * counter))
        # sys.stdout.write('%s]' % (' ' * (toolbar_width - counter)))
        # sys.stdout.flush()
        # counter += t_value

    plt.plot(t_values, ada_boost_train_error, label='Ada Test')
    plt.plot(t_values, ada_boost_test_error, label='Ada Train')
    plt.plot(t_values, bagging_trees_train_error, label='Bag Test')
    plt.plot(t_values, bagging_trees_test_error, label='Bag Train')
    plt.plot(t_values, random_forests_train_error, label='Forest Test')
    plt.plot(t_values, random_forests_test_error, label='Forest Train')
    plt.legend(loc='best')
    plt.show()


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
