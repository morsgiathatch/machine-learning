from DecisionTree import Id3
from Data.bank import BankData
import os
from Tests.DecisionTreeTests import non_numeric_id3_test


def numeric_id3_test():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    metrics = {0: 'information_gain', 1: 'majority_error_gain', 2: 'gini_index_gain'}

    # Begin prompts
    unknown_choice = input("Do you wish to use 'unknown' as an attribute? y/n ")

    if unknown_choice == "n":
        use_unknown = True
    else:
        use_unknown = False

    use_averages = input("Would you like to calculate averages over all metrics? y/n ")

    # Construct tree
    data = BankData.Data()
    data.initialize_data_from_file(dir_path + '/../Data/bank/train.csv', use_unknown)

    # Test tree
    test_data = BankData.Data()
    test_data.initialize_data_from_file(dir_path + '/../Data/bank/test.csv', use_unknown)
    if use_averages == "y":
        non_numeric_id3_test.calculate_averages(data, test_data, metrics, 17)

    else:
        tree_depth = int(input("Please enter desired tree depth [1 - 16] (0 to run entire tree):"))
        if tree_depth == 0:
            tree_depth = float("inf")

        metric_choice = int(input("Please enter a number for choice of metric:\n0: Information Gain\n"
                                  "1. Majority Error\n2. Gini Index\n"))
        metric = metrics[metric_choice]

        # Test for noise in data
        print("Begin detecting noise")
        noise_count = 0
        for example in data.examples:
            for other_example in data.examples:
                if example == other_example:
                    if example.label != other_example.label:
                        noise_count += 1

        print("Detected " + str(noise_count) + " features of noise")

        # Begin fit
        height = run_id3(data, test_data, metric, tree_depth, None, None)
        print("max tree height: " + str(height))


def run_id3(data, test_data, metric, tree_depth, data_percents, train_data_percents):
    id3 = Id3.Id3(metric)
    print("\n--- Using Tree level " + str(tree_depth) + " ---")
    id3.fit(data.examples, data.attributes, None, data.labels, 0, tree_depth)

    correct_results = 0
    for example in test_data.examples:
        if example.get_label() == id3.predict(example):
            correct_results += 1

    percentage = float(correct_results) / float(len(test_data.examples))
    if data_percents is not None:
        data_percents.append(percentage)

    print("Test Error: " + "%.16f" % (1.0 - percentage))

    correct_results = 0
    for example in data.examples:
        if example.get_label() == id3.predict(example):
            correct_results += 1

    percentage = float(correct_results) / float(len(data.examples))
    if train_data_percents is not None:
        train_data_percents.append(percentage)

    print("Training Error: " + "%.16f" % (1.0 - percentage))
    max_height = id3.max_height
    id3.reset_max_height()

    return max_height
