from DecisionTree import BankData
from EnsembleLearning import BaggingTrees
import matplotlib.pyplot as plt
import os

def problem2b():
    # Construct tree
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankData.Data()
    data.initialize_data_from_file(dir_path + '/../Data/bank/train.csv', False)

    # Test tree
    test_data = BankData.Data()
    test_data.initialize_data_from_file(dir_path + '/../Data/bank/test.csv', False)

    test_percentages = []
    train_percentages = []

    t_values = [1, 2, 5, 10, 25, 50, 75, 125, 250, 500, 1000]
    # t_values = [1, 2, 5, 10]
    for t_value in t_values:
        print("\nRunning Bagging Trees on " + str(t_value) + " Trees\n")
        trees = BaggingTrees.run_bagging_trees(t_value, data)

        correct_results = 0
        for example in test_data.examples:
            if example.get_label() == BaggingTrees.get_result(example, trees, data):
                correct_results += 1

        percentage = float(correct_results) / float(len(test_data.examples))

        test_percentages.append(1.0 - percentage)

        print("Test Error: " + "%.16f" % (1.0 - percentage))

        correct_results = 0
        for example in data.examples:
            if example.get_label() == BaggingTrees.get_result(example, trees, test_data):
                correct_results += 1

        percentage = float(correct_results) / float(len(data.examples))

        train_percentages.append(1.0 - percentage)

        print("Train Error: " + "%.16f" % (1.0 - percentage))

    plt.plot(t_values, test_percentages, label='Test')
    plt.plot(t_values, train_percentages, label='Train')
    plt.legend(loc='best')
    plt.show()
