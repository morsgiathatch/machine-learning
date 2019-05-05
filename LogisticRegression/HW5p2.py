from Perceptron import BankNoteData
from NeuralNetworks import GradientDescent
from LogisticRegression import LogisticRegressors as lr
import numpy as np
import matplotlib.pyplot as plt
import os


def hw5p2():
    redo_problem = True

    while redo_problem:
        problem = int(input("\nPlease choose a problem\n1. Problem a\n2. Problem b\n3. Exit\n"))
        valid_choice = True
        if problem != 1 and problem != 2 and problem != 3:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a problem\n1. Problem a\n2. Problem b\n3. Exit\n"))
            if problem == 1 or problem == 2 or problem == 3:
                valid_choice = True

        if problem == 1:
            hw5p2a()
        elif problem == 2:
            hw5p2b()
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW5 Problem 2? y/n\n"))
        if should_redo == "n":
            redo_problem = False


def hw5p2a():
    helper(parta=True, obj_fun=lr.map_estimator, grad_func=lr.map_estimator_gradient)


def hw5p2b():
    helper(parta=False, obj_fun=lr.ml_estimator, grad_func=lr.ml_estimator_gradient)


def helper(parta, obj_fun, grad_func):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/train.csv', shift_origin=True, )
    test_data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/test.csv', shift_origin=True)

    if parta:
        variances = np.array([0.01, 0.1, 0.5, 1, 3, 5, 10, 100])
    else:
        variances = [0]
    max_iters = 100
    train_percentages = []
    test_percentages = []
    for var in variances:
        grad_descent = GradientDescent.GradientDescent(features=data.features, labels=data.output, gamma_0=0.001, d=100.0)
        results = grad_descent.run_stochastic_sub_grad_descent(max_iters=max_iters, obj_func=obj_fun,
                                                               grad_func=grad_func,
                                                               weights=np.zeros(data.features.shape[1]), args=var)

        train_percentage_ = lr.get_percentages(data, results[0])
        test_percentage_ = lr.get_percentages(test_data, results[0])
        train_percentages.append(train_percentage_)
        test_percentages.append(test_percentage_)
        print("Train error percentage was %.16f" % train_percentage_)
        print("Test error percentage was %.16f" % test_percentage_)

        t = np.linspace(0, max_iters, max_iters)
        plt.plot(t, results[1], label='Objective Function')
        plt.legend()
        plt.show()

    if parta:
        plt.semilogx(variances, train_percentages, label='Train Percentages')
        plt.semilogx(variances, test_percentages, label='Test Percentages')
        plt.legend()
        plt.show()
