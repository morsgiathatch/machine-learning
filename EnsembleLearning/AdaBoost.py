from DecisionTree import Id3
from DecisionTree import Metrics
import numpy as np


def run_Adaboost(data, test_data, t_value):
    print("\nRunning AdaBoost on " + str(t_value) + " iterations.\n")
    dt = []
    for example in data.examples:
        dt.append(float(1/len(data.examples)))
    alphas = []
    h_classifiers = []
    for example in data.examples:
        example.set_weight(float(1/len(data.examples)))

    # epsilons = []

    for i in range(0, t_value):
        id3 = Id3.Id3()
        root = id3.id3(data.examples, data.attributes, None, data.labels, 0, 1, Metrics.weighted_information_gain)
        h_classifiers.append(root)
        # Get predictions
        h_predictions = []
        for example in data.examples:
            h_predictions.append(data.get_test_result(example, root))

        epsilon = get_epsilon(data, h_predictions, dt)
        alphas.append(0.5 * np.log((1.0 - epsilon) / epsilon))
        # epsilons.append(epsilon)

        update_dt(dt, alphas[i], data, h_predictions)

        j = 0
        for example in data.examples:
            example.set_weight(dt[j])
            j += 1

    percentages = []

    correct_results = 0
    for example in test_data.examples:
        if example.get_label() == get_final_hypothesis(t_value, alphas, example, h_classifiers, test_data):
            correct_results += 1

    percentage = float(correct_results) / float(len(test_data.examples))

    percentages.append(percentage)

    print("Test Error: " + "%.16f" % (1.0 - percentage))

    correct_results = 0
    for example in data.examples:
        if example.get_label() == get_final_hypothesis(t_value, alphas, example, h_classifiers, data):
            correct_results += 1

    percentage = float(correct_results) / float(len(data.examples))

    print("Train Error: " + "%.16f" % (1.0 - percentage))

    percentages.append(percentage)
    return percentages


def get_epsilon(data, h_predictions, dt):
    i = 0
    sum = 0.0
    for example in data.examples:
        sum += dt[i] * example.get_label() * h_predictions[i]
        i += 1

    return 0.5 - (0.5 * sum)


def update_dt(dt, alpha, data, h_predictions):

    i = 0
    for example in data.examples:
        dt[i] *= np.exp(-1.0 * alpha * example.get_label() * h_predictions[i])
        i += 1

    # Get total sum and normalize over all weights
    sum = 0.0
    for i in range(0, len(dt)):
        sum += dt[i]

    for i in range(0, len(dt)):
        dt[i] /= sum


def get_final_hypothesis(t_value, alphas, example, h_classifiers, data):
    sum = 0.0
    for i in range(0, t_value):
        sum += alphas[i] * data.get_test_result(example, h_classifiers[i])

    return np.sign(sum)
