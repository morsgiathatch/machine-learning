from NeuralNetworks import ThreeLayerNN
from Data.bank_note import BankNoteData
from Algorithms import GradientDescent
import numpy as np
import matplotlib.pyplot as plt
import os


def neural_network_test():
    redo_test = True

    while redo_test:
        test_choice = int(input("\nPlease choose a test\n1. Weights Initialized From Standard Normal Distribution\n"
                                "2. Weights Initialized to Zero\n3. Exit\n"))
        valid_choice = True
        if test_choice not in range(1, 4):
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            test_choice = int(input("\nPlease choose a test\n1. Weights Initialized From Standard Normal Distribution\n"
                                    "2. Weights Initialized to Zero\n3. Exit\n"))
            if test_choice in range(1, 4):
                valid_choice = True

        if test_choice == 2:
            stand_norm_weights()
        elif test_choice == 3:
            zero_weights()
        else:
            break

        should_redo = str(input("\nWould you like to run another Neural Network test? y/n\n"))
        if should_redo == "n":
            redo_test = False


def stand_norm_weights():
    helper(partb=True)


def zero_weights():
    helper(partb=False)


def helper(partb):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/train.csv', shift_origin=True)
    test_data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/test.csv', shift_origin=True)

    layer_widths = [5, 10, 25, 50, 100]
    train_percentages = []
    test_percentages = []
    max_iters = 5
    for layer_width in layer_widths:
        print("Running test width hidden layer width: %i" % layer_width)
        if partb:
            weights = np.zeros((layer_width, layer_width, 4))
            for i in range(0, layer_width):
                for j in range(0, layer_width):
                    weights[i, j, 2] = np.random.normal()

            for i in range(0, layer_width):
                weights[i, 1, 3] = np.random.normal()

            for i in range(0, data.features.shape[1]):
                for j in range(0, layer_width):
                    weights[i, j, 1] = np.random.normal()

        else:
            weights = np.zeros((layer_width, layer_width, 4))

        three_layer_nn = ThreeLayerNN.ThreeLayerNN(num_units_per_layer=layer_width, weights=weights)
        grad_descent = GradientDescent.GradientDescent(features=data.features, labels=data.output)
        results = grad_descent.fit_stochastic(max_iters=max_iters, args=None, weights=weights,
                                              step_function=training_schedule,
                                              obj_func=three_layer_nn.objective_function,
                                              grad_func=three_layer_nn.gradient)

        train_percentage = get_percentages(data, three_layer_nn.predict)
        test_percentage = get_percentages(test_data, three_layer_nn.predict)
        train_percentages.append(train_percentage)
        test_percentages.append(test_percentage)
        print("Train error percentage was %.16f" % train_percentage)
        print("Test error percentage was %.16f" % test_percentage)

        t = np.linspace(0, max_iters, max_iters)
        plt.plot(t, results[1])
        plt.show()

    plt.semilogx(layer_widths, train_percentages, label='Train Percentages')
    plt.semilogx(layer_widths, test_percentages, label='Test Percentages')
    plt.legend()
    plt.show()


def get_percentages(data, prediction):
    num_correct = 0
    for row_ndx in range(0, data.features.shape[0]):
        if data.output[row_ndx] == np.sign(prediction(data.features[row_ndx, :])):
            num_correct += 1

    return 1.0 - float(num_correct / data.features.shape[0])


def training_schedule(t):
    gamma_0 = 0.5
    d = 0.1
    return gamma_0 / (1.0 + (gamma_0 / d) * t)
