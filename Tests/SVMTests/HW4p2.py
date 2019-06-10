import numpy as np
from Algorithms import GradientDescent
from Data.bank_note import BankNoteData
from Perceptron import Perceptron
from Tests.PerceptronTests import HW3p2
from SVM import SVM
import matplotlib.pyplot as plt
from numpy import linalg as la
import os
import sys


def hw4p2():
    redo_problem = True

    while redo_problem:

        problem = int(input("\nPlease choose a problem part\n1. "
                            "Problem 2a\n2. Problem 2b\n3. Problem 2c\n4. Exit\n"))

        valid_choice = True
        if problem != 1 and problem != 2 and problem != 3 and problem != 4:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a problem part4\n1. "
                                "Problem 2a\n2. Problem 2b\n3. Problem 2c\n4. Exit\n"))

            if problem == 1 or problem == 2 or problem == 3 or problem == 4:
                valid_choice = True

        C_values = np.array([1., 2., 50., 100., 300., 500., 700.])
        C_values = (1.0 / 873. * C_values).tolist()

        if problem == 1:
            choice_a_or_b('a', print_progress=True, num_reps=10, C_values=C_values, show_plots=True)
        elif problem == 2:
            choice_a_or_b('b', print_progress=True, num_reps=10, C_values=C_values, show_plots=True)
        elif problem == 3:
            choice_c(C_values)
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW 4 problem 2? y/n\n"))
        if should_redo == "n":
            redo_problem = False


def choice_c(C_values):
    num_reps = int(input("How many iterates would you like to use to calculate averages?\n"))
    print("Calculating results from first schedule")
    [w_vectors_from_a, train_pcts_from_a, test_pcts_from_a] = choice_a_or_b('a', print_progress=False, num_reps=num_reps, C_values=C_values, show_plots=False)
    print("Calculating results from second schedule")
    [w_vectors_from_b, train_pcts_from_b, test_pcts_from_b] = choice_a_or_b('b', print_progress=False, num_reps=num_reps, C_values=C_values, show_plots=False)

    # plot differences of norms and absolute training/testing errors by C
    norms = []
    abs_train_diffs = []
    abs_test_diffs = []
    for i in range(0, len(w_vectors_from_a)):
        norms.append(la.norm((w_vectors_from_a[i] / la.norm(w_vectors_from_a[i])) - (w_vectors_from_b[i] / la.norm(w_vectors_from_b[i]))))
        abs_train_diffs.append(np.abs(train_pcts_from_a[i] - train_pcts_from_b[i]))
        abs_test_diffs.append(np.abs(test_pcts_from_a[i] - test_pcts_from_b[i]))

    plt.plot(C_values, np.array(norms), C_values, np.array(abs_train_diffs), C_values, np.array(abs_test_diffs))
    plt.xlabel('C values')
    plt.ylabel('Errors')
    plt.legend(('Norm error', 'Train Error', 'Test Error'))
    plt.show()


def choice_a_or_b(part, print_progress, num_reps, C_values, show_plots):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/train.csv', shift_origin=True)

    test_data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/test.csv', shift_origin=True)

    test_percentages = []
    train_percentages = []

    w_vectors = []

    training_schedule = training_schedule_b
    if part == 'a':
        training_schedule = training_schedule_a

    count = 0
    for i, C_value in enumerate(C_values):
        if print_progress:
            print("\nRunning test with C = %f" % C_value)

        train_percentages_per_C = []
        test_percentages_per_C = []

        w_vectors_by_C = []

        for j in range(0, num_reps):
            count += 1
            grad_desc = GradientDescent.GradientDescent(data.features, data.output)
            [w_vector, obj_func_vals] = grad_desc.fit_stochastic(max_iters=100,
                                                                 obj_func=SVM.objective_function,
                                                                 grad_func=SVM.stoch_grad_func,
                                                                 args=[C_value, data.features.shape[0]],
                                                                 step_function=training_schedule,
                                                                 print_status=False)

            w_vectors_by_C.append(w_vector)

            if print_progress:
                sys.stdout.write('\r%i / 10' % (j + 1))
                sys.stdout.flush()
            else:
                sys.stdout.write('\rProgress: %i / %i' % (count, len(C_values) * num_reps))
                sys.stdout.flush()

            if j == 0 and show_plots:
                plt.plot(np.linspace(0, 100, 100), obj_func_vals)
                plt.show()

        # Compute average vector
        w_vectors_by_C = np.array(w_vectors_by_C)
        avg_w_vector = w_vectors_by_C.mean(0)

        train_percentages_per_C.append(HW3p2.get_percentages(avg_w_vector, data, Perceptron.get_prediction))
        test_percentages_per_C.append(HW3p2.get_percentages(avg_w_vector, test_data, Perceptron.get_prediction))
        w_vectors.append(avg_w_vector)

        if print_progress:
            sys.stdout.write('\n')
            sys.stdout.flush()

        train_percentage = np.average(np.array(train_percentages_per_C))
        test_percentage = np.average(np.array(test_percentages_per_C))
        train_percentages.append(train_percentage)
        test_percentages.append(test_percentage)

        if print_progress:
            print("Train error percentage with average vector was %.16f" % train_percentage)
            print("Test error percentage with average vector was %.16f" % test_percentage)

    train_percentages = np.array(train_percentages)
    test_percentages = np.array(test_percentages)

    if print_progress:
        plt.plot(C_values, train_percentages, C_values, test_percentages)
        plt.xlabel('C values')
        plt.ylabel('Errors')
        plt.legend(('Train Error', 'Test Error'))
        plt.show()
    else:
        sys.stdout.write('\n')
        sys.stdout.flush()

    return [w_vectors, train_percentages, test_percentages]


def training_schedule_a(t):
    gamma_0 = 0.5
    d = 0.1
    return gamma_0 / (1 + (gamma_0 / d) * t)


def training_schedule_b(t):
    gamma_0 = 0.5
    return gamma_0 / (1 + t)
