from Data.bank import BankData
from EnsembleLearning import AdaBoost
import matplotlib.pyplot as plt
import numpy as np
import os


def problem2a():
    # Construct tree
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankData.Data()
    data.initialize_data_from_file(dir_path + '/../../Data/bank/train.csv', False)

    # Test tree
    test_data = BankData.Data()
    test_data.initialize_data_from_file(dir_path + '/../../Data/bank/test.csv', False)

    t_values = [1, 2, 5, 10, 25, 50, 75, 125, 250, 500, 1000]
    # t_values = [1, 2, 5, 10]

    train_percentages = []
    test_percentages = []
    decision_stumps = []
    for t_value in t_values:
        adaboost = AdaBoost.Adaboost(features=data.examples, attributes=data.attributes, t_value=t_value)
        adaboost.fit(print_status=True)
        percentages = get_percentages(adaboost, test_data, data)

        test_percentages.append(percentages[0])
        train_percentages.append(percentages[1])
        if t_value == 1000:
            decision_stumps = adaboost.h_classifiers

    plt.plot(t_values, test_percentages, label='Test')
    plt.plot(t_values, train_percentages, label='Train')
    plt.legend(loc='best')
    plt.show()

    stump_train_percentages = []
    stump_test_percentages = []
    for stump in decision_stumps:
        correct_results = 0
        for example in test_data.examples:
            if example.get_label() == stump.predict(example):
                correct_results += 1

        percentage = float(correct_results) / float(len(test_data.examples))

        stump_test_percentages.append(1.0 - percentage)

        correct_results = 0
        for example in data.examples:
            if example.get_label() == stump.predict(example):
                correct_results += 1

        percentage = float(correct_results) / float(len(data.examples))

        stump_train_percentages.append(1.0 - percentage)

    x_vals = np.linspace(0, 1000, num=1000)

    plt.plot(x_vals, stump_test_percentages, label='Test')
    plt.plot(x_vals, stump_train_percentages, label='Train')
    plt.legend(loc='best')
    plt.show()


def get_percentages(adaboost, test_data, data):
    percentages = []

    correct_results = 0
    for example in test_data.examples:
        if example.get_label() == adaboost.predict(example):
            correct_results += 1

    percentage = float(correct_results) / float(len(test_data.examples))

    percentages.append(1.0 - percentage)

    print("Test Error: " + "%.16f" % (1.0 - percentage))

    correct_results = 0
    for example in data.examples:
        if example.get_label() == adaboost.predict(example):
            correct_results += 1

    percentage = float(correct_results) / float(len(data.examples))

    print("Train Error: " + "%.16f" % (1.0 - percentage))

    percentages.append(1.0 - percentage)

    return percentages
