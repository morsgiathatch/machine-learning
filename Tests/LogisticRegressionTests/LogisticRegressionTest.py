from Data.bank_note import BankNoteData
from Algorithms import GradientDescent
from LogisticRegression import LogisticRegressors as lr
import numpy as np
import matplotlib.pyplot as plt
import os


def logistic_regression_test():
    redo_test = True

    while redo_test:
        test_choice = int(input("\nPlease choose a test\n1. MAP Estimator Test\n2. ML Estimator Test\n3. Exit\n"))
        valid_choice = True
        if test_choice not in range(1, 4):
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            test_choice = int(input("\nPlease choose a test\n1. MAP Estimator Test\n2. ML Estimator Test\n3. Exit\n"))
            if test_choice in range(1, 4):
                valid_choice = True

        if test_choice == 1:
            map_estimator_test()
        elif test_choice == 2:
            ml_estimator_test()
        else:
            break

        should_redo = str(input("\nWould you like to run another Logistic Regression test? y/n\n"))
        if should_redo == "n":
            redo_test = False


def map_estimator_test():
    helper(parta=True, obj_fun=lr.map_estimator, grad_func=lr.map_estimator_gradient)


def ml_estimator_test():
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
        grad_descent = GradientDescent.GradientDescent(features=data.features, labels=data.output)
        results = grad_descent.fit_stochastic(max_iters=max_iters, obj_func=obj_fun,
                                              grad_func=grad_func, step_function=training_schedule,
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


def training_schedule(t):
    gamma_0 = 0.001
    d = 100.0
    return gamma_0 / (1.0 + (gamma_0 / d) * t)
