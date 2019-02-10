import Id3
import CarData
import Metrics
import sys
import os


def problem1():
    # Construct tre
    data = CarData.Data()
    data.initialize_data_from_file(os.getcwd() + "/DecisionTree/car/train.csv")
    metrics = {0: Metrics.information_gain, 1: Metrics.majority_error_gain, 2: Metrics.gini_index_gain}

    # Test tree
    test_data = CarData.Data()
    test_data.initialize_data_from_file(os.getcwd() + "/DecisionTree/car/test.csv")

    # Detect Noise
    print "Begin detecting noise"
    noise_count = 0
    for example in data.examples:
        for other_example in data.examples:
            if example == other_example:
                if example.label != other_example.label:
                    noise_count += 1

    print "Detected " + str(noise_count) + " examples of noise\n"

    # Begin prompts

    use_averages = raw_input("Would you like to calculate averages over all metrics? y/n ")

    if use_averages == "y":
        calculate_averages(data, test_data, metrics)

    else:
        tree_depth = int(raw_input("Please enter desired tree depth [1 - 6] (0 to run entire tree):"))
        if tree_depth == 0:
            tree_depth = sys.maxint

        metric_choice = int(raw_input("Please enter a number for choice of metric:\n0: Information Gain\n"
                                  "1. Majority Error\n2. Gini Index\n"))
        metric = metrics[metric_choice]

        # Run ID3
        run_id3(data, test_data, metric, tree_depth, None, None)


def run_id3(data, test_data, metric, tree_depth, data_percents, train_data_percents):
    id3 = Id3.Id3()
    print "\n--- Using Tree level " + str(tree_depth) + " ---"
    root = id3.id3(data.examples, data.attributes, None, data.labels, 0, tree_depth, metric)

    correct_results = 0
    for example in test_data.examples:
        if example.get_label() == test_data.get_test_result(example, root):
            correct_results += 1

    percentage = float(correct_results) / float(len(test_data.examples))
    if data_percents is not None:
        data_percents.append(percentage)

    print "Test Error: " + str(1.0 - percentage)

    correct_results = 0
    for example in data.examples:
        if example.get_label() == data.get_test_result(example, root):
            correct_results += 1

    percentage = float(correct_results) / float(len(data.examples))
    if train_data_percents is not None:
        train_data_percents.append(percentage)

    print "Training Error: " + str(1.0 - percentage)
    max_height = id3.max_height
    id3.reset_max_height()

    return max_height


def calculate_averages(data, test_data, metrics):
    information_gains = []
    information_gains_train = []
    max_errors = []
    max_errors_train = []
    ginis = []
    ginis_train = []
    values = [information_gains, max_errors, ginis]
    values_train = [information_gains_train, max_errors_train, ginis_train]
    metric_names = {0: " Information Gain ", 1: " Majority Error ", 2: " Gini Index "}

    max_j = 0
    for i in range(0, 3):
        print "\n------------- " + metric_names[i] + " -------------\n"
        for j in range(1, 7):
            max_height = run_id3(data, test_data, metrics[i], j, values[i], values_train[i])
            #print "Height of tree is " + str(max_height)
            if max_height < j:
                max_j = j
                break

    # Pop last value since it is a duplicate and we can no longer grow the tree
    if max_j != 0:
        values.pop()
        values_train.pop()

    # Calculate and print averages
    print "\n-- Test data average for metrics --"
    print "Information gain: " + str(1.0 - average(values[0]))
    print "Majority Error: " + str(1.0 - average(values[1]))
    print "Gini Index: " + str(1.0 - average(values[2]))
    print "\n-- Train data average for metrics --"
    print "Information gain: " + str(1.0 - average(values_train[0]))
    print "Majority Error: " + str(1.0 - average(values_train[1]))
    print "Gini Index: " + str(1.0 - average(values_train[2]))


def average(data):
    length = len(data)
    temp_sum = 0.0
    for element in data:
        temp_sum += element

    return temp_sum / float(length)
