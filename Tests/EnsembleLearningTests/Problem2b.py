from Data.bank import BankData
from EnsembleLearning import BaggingTrees
import matplotlib.pyplot as plt
import os


def problem2b():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankData.Data()
    data.initialize_data_from_file(dir_path + '/../../Data/bank/train.csv', False)

    test_data = BankData.Data()
    test_data.initialize_data_from_file(dir_path + '/../../Data/bank/test.csv', False)

    test_percentages = []
    train_percentages = []

    factor = float(input("Please enter a number to get fraction of examples. (e.g., `2` uses 1/2 of examples):\n"))

    t_values = [1, 2, 5, 10, 25, 50, 75, 125, 250, 500, 1000]
    # t_values = [1, 2, 5, 10]
    for t_value in t_values:
        print("\nRunning Bagging Trees on " + str(t_value) + " Trees\n")
        bagging_trees = BaggingTrees.BaggingTrees(t_value=t_value, features=data.examples, attributes=data.attributes,
                                                  attribute_factor=factor)
        bagging_trees.fit(print_status_bar=False)

        correct_results = 0
        for example in test_data.examples:
            if example.get_label() == bagging_trees.predict(example):
                correct_results += 1

        percentage = float(correct_results) / float(len(test_data.examples))

        test_percentages.append(1.0 - percentage)

        print("Test Error: " + "%.16f" % (1.0 - percentage))

        correct_results = 0
        for example in data.examples:
            if example.get_label() == bagging_trees.predict(example):
                correct_results += 1

        percentage = float(correct_results) / float(len(data.examples))

        train_percentages.append(1.0 - percentage)

        print("Train Error: " + "%.16f" % (1.0 - percentage))

    plt.plot(t_values, test_percentages, label='Test')
    plt.plot(t_values, train_percentages, label='Train')
    plt.legend(loc='best')
    plt.show()
