from DecisionTree import BankData
from EnsembleLearning import AdaBoost
import matplotlib.pyplot as plt
import os

def problem2a():
    # Construct tree
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankData.Data()
    data.initialize_data_from_file(dir_path + '/../Data/bank/train.csv', False)

    # Test tree
    test_data = BankData.Data()
    test_data.initialize_data_from_file(dir_path + '/../Data/bank/test.csv', False)

    t_values = [1, 2, 5, 10, 25, 50, 75, 125, 250, 500, 1000]
    train_percentages = []
    test_percentages = []
    for t_value in t_values:
        results = AdaBoost.run_Adaboost(data, test_data, t_value)
        test_percentages.append(results[0])
        train_percentages.append(results[1])

    plt.plot(t_values, test_percentages, label='Test')
    plt.plot(t_values, train_percentages, label='Train')
    plt.legend(loc='best')
    plt.show()
