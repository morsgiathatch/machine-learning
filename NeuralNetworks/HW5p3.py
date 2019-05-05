from NeuralNetworks import GradientDescent
from NeuralNetworks import ThreeLayerNN
from Perceptron import BankNoteData
import numpy as np
import matplotlib.pyplot as plt
import os


def hw5p3():
    redo_problem = True

    while redo_problem:
        problem = int(input("\nPlease choose a problem\n2. Problem b\n3. Problem c\n4. Exit\n"))
        valid_choice = True
        if problem != 2 and problem != 3 and problem != 4:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a problem\n2. Problem b\n3. Problem c\n4. Exit\n"))
            if problem == 2 or problem == 3 or problem == 4:
                valid_choice = True

        if problem == 2:
            hw5pb()
        elif problem == 3:
            hw5pc()
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW5 Problem 3? y/n\n"))
        if should_redo == "n":
            redo_problem = False


def hw5pb():
    helper(partb=True)

def hw5pc():
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
        grad_descent = GradientDescent.GradientDescent(features=data.features, labels=data.output, gamma_0=0.5, d=0.1)
        results = grad_descent.run_stochastic_sub_grad_descent(max_iters=max_iters, args=None,
                                                               obj_func=three_layer_nn.objective_function,
                                                               grad_func=three_layer_nn.gradient, weights=weights)

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
