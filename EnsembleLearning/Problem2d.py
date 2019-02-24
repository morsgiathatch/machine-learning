from DecisionTree import BankData
from EnsembleLearning import RandomForests
import matplotlib.pyplot as plt
import os


def problem2d():
    # Train data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankData.Data()
    data.initialize_data_from_file(dir_path + '/../Data/bank/train.csv', False)

    # Test data
    test_data = BankData.Data()
    test_data.initialize_data_from_file(dir_path + '/../Data/bank/test.csv', False)

    t_values = [1, 2, 5, 10, 25, 50, 75, 125, 250, 500, 1000]
    # t_values = [1, 2, 5, 10]
    attr_sizes = [2, 4, 6]
    test_results = []
    train_results = []

    for size in attr_sizes:
        test_percentages = []
        train_percentages = []

        for t_value in t_values:
            print("\nRunning Random Forests on " + str(t_value) + " Trees with Attribute Size " + str(size) + "\n")
            forest = RandomForests.runRandomForests(t_value, data.examples, data.attributes, data.labels, size)

            correct_results = 0
            for example in test_data.examples:
                if example.get_label() == RandomForests.get_result(example, forest, data):
                    correct_results += 1

            percentage = float(correct_results) / float(len(test_data.examples))

            test_percentages.append(1.0 - percentage)

            print("Test Error: " + "%.16f" % (1.0 - percentage))

            correct_results = 0
            for example in data.examples:
                if example.get_label() == RandomForests.get_result(example, forest, test_data):
                    correct_results += 1

            percentage = float(correct_results) / float(len(data.examples))

            train_percentages.append(1.0 - percentage)

            print("Train Error: " + "%.16f" % (1.0 - percentage))

        test_results.append(test_percentages)
        train_results.append(train_percentages)

    for i in range(0, len(attr_sizes)):
        plt.plot(t_values, test_results[i], label='Test: Attr. Size = ' + str(attr_sizes[i]))
        plt.plot(t_values, train_results[i], label='Train: Attr. Size = ' + str(attr_sizes[i]))

    plt.legend(loc='best')
    plt.show()
