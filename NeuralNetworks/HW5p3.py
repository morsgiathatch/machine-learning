from NeuralNetworks import GradientDescent
from NeuralNetworks import ThreeLayerNN
from Perceptron import BankNoteData
import numpy as np
import matplotlib.pyplot as plt
import os


def hw5p3():
    redo_problem = True

    while redo_problem:
        problem = int(input("\nPlease choose a problem\n2. Problem b\n3. Problem c\n4. Problem d\n5. Exit\n"))
        valid_choice = True
        if problem != 2 and problem != 3 and problem != 4 and problem != 5:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a problem\n2. Problem b\n3. Problem c\n4. Problem d\n5. Exit\n"))
            if problem == 2 or problem == 3 or problem == 4 or problem == 5:
                valid_choice = True

        if problem == 2:
            hw5pb()
        elif problem == 3:
            hw5pc()
        elif problem == 4:
            hw5pd()
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW5? y/n\n"))
        if should_redo == "n":
            redo_problem = False


def hw5pb():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/train.csv', shift_origin=True)
    test_data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/test.csv', shift_origin=True)

    weights = np.zeros((data.features.shape[1], data.features.shape[1], 4))
    for i in range(0, data.features.shape[1]):
        for j in range(0, data.features.shape[1]):
            for k in range(0, 3):
                weights[i, j, k] = np.random.normal()

    three_layer_nn = ThreeLayerNN.ThreeLayerNN(num_units_per_layer=data.features.shape[1], weights=weights)
    grad_descent = GradientDescent.GradientDescent(features=data.features, labels=data.output, gamma_0=0.1, d=1.0)
    grad_descent.run_stochastic_sub_grad_descent(max_iters=1, obj_func=three_layer_nn.objective_function,
                                                 grad_func=three_layer_nn.gradient, weights=weights)
    train_percentage = get_percentages(data, three_layer_nn.predict)
    test_percentage = get_percentages(test_data, three_layer_nn.predict)
    print("Train error percentage was %.16f" % train_percentage)
    print("Test error percentage was %.16f" % test_percentage)

    t = np.linspace(0, data.features.shape[0], data.features.shape[0])
    # plt.plot(t, results[1])

def hw5pc():
    return None


def hw5pd():
    return None


def get_percentages(data, prediction):
    num_correct = 0
    for row_ndx in range(0, data.features.shape[0]):
        if data.output[row_ndx] == np.sign(prediction(data.features[row_ndx, :])):
            num_correct += 1

    return 1.0 - float(num_correct / data.features.shape[0])
