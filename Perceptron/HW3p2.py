import os
from Perceptron import BankNoteData
from Perceptron import Perceptron
import numpy as np
import sys


def hw3p2():
    redo_problem = True

    while redo_problem:

        problem = int(input("\nPlease choose a problem part4\n1. "
                            "Problem 2a\n2. Problem 2b\n3. Problem 2c\n4. Problem 2d\n5. Exit\n"))

        valid_choice = True
        if problem != 1 and problem != 2 and problem != 3 and problem != 4 and problem != 5:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a problem part4\n1. "
                                "Problem 2a\n2. Problem 2b\n3. Problem 2c\n4. Problem 2d\n5. Exit\n"))

            if problem == 1 or problem == 2 or problem == 3 or problem == 4 or problem == 5:
                valid_choice = True

        if problem == 1:
            choice_a()
        elif problem == 2:
            choice_b()
        elif problem == 3:
            choice_c()
        elif problem == 4:
            choice_d()
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW 3 problem 2? y/n\n"))
        if should_redo == "n":
            redo_problem = False


def choice_a():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/train.csv')

    test_data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/test.csv')

    perceptron = Perceptron.perceptron(10, data.features, data.output, 0.5)

    train_percentage = get_percentages(perceptron, data, Perceptron.get_prediction)
    test_percentage = get_percentages(perceptron, test_data, Perceptron.get_prediction)

    print("Weight vector was:")
    print(perceptron)
    print("Train error percentage was %.16f" % train_percentage)
    print("Test error percentage was %.16f" % test_percentage)


def choice_b():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/train.csv')

    test_data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/test.csv')

    perceptron = Perceptron.voted_perceptron(10, data.features, data.output, 0.5)

    print("The vectors and their respective count are:")
    np.set_printoptions(precision=4)
    for i in range(0, len(perceptron[0])):
        print(perceptron[0][i], "  : %s" % (str(perceptron[1][i])), )

    train_percentage = get_percentages(perceptron, data, Perceptron.get_voted_prediction)
    test_percentage = get_percentages(perceptron, test_data, Perceptron.get_voted_prediction)

    print("Train error percentage was %.16f" % train_percentage)
    print("Test error percentage was %.16f" % test_percentage)


def choice_c():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/train.csv')

    test_data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/test.csv')

    perceptron = Perceptron.averaged_perceptron(10, data.features, data.output, 0.5)

    train_percentage = get_percentages(perceptron, data, Perceptron.get_prediction)
    test_percentage = get_percentages(perceptron, test_data, Perceptron.get_prediction)

    print("Weight vector was:")
    print(perceptron)
    print("Train error percentage was %.16f" % train_percentage)
    print("Test error percentage was %.16f" % test_percentage)


def choice_d():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/train.csv')
    test_data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/test.csv')

    train_errors = [[], [], []]
    test_errors = [[], [], []]

    print("Calculating statistics")
    sys.stdout.write("Progress: [%s]" % (" " * 100))
    sys.stdout.flush()
    for i in range(1, 101):
        standard_perceptron = Perceptron.perceptron(10, data.features, data.output, 0.5)
        voted_perceptron = Perceptron.voted_perceptron(10, data.features, data.output, 0.5)
        averaged_perceptron = Perceptron.averaged_perceptron(10, data.features, data.output, 0.5)

        train_errors[0].append(get_percentages(standard_perceptron, data, Perceptron.get_prediction))
        train_errors[1].append(get_percentages(voted_perceptron, data, Perceptron.get_voted_prediction))
        train_errors[2].append(get_percentages(averaged_perceptron, data, Perceptron.get_prediction))

        test_errors[0].append(get_percentages(standard_perceptron, test_data, Perceptron.get_prediction))
        test_errors[1].append(get_percentages(voted_perceptron, test_data, Perceptron.get_voted_prediction))
        test_errors[2].append(get_percentages(averaged_perceptron, test_data, Perceptron.get_prediction))

        sys.stdout.write('\r')
        sys.stdout.flush()
        sys.stdout.write('Progress: [%s' % ('#' * i))
        sys.stdout.write('%s]' % (' ' * (100 - i)))
        sys.stdout.flush()

    print("\n%24s   %10s   %10s" % ("Standard", "Voted", "Averaged"))
    print("Train Errors: %.8f   %.8f   %.8f" % (float(np.mean(np.array(train_errors[0]))),
                                                float(np.mean(np.array(train_errors[1]))),
                                                float(np.mean(np.array(train_errors[2])))))
    print("Test Errors:  %.8f   %.8f   %.8f" % (float(np.mean(np.array(test_errors[0]))),
                                                float(np.mean(np.array(test_errors[1]))),
                                                float(np.mean(np.array(test_errors[2])))))


def get_percentages(perceptron, data, prediction):
    num_correct = 0
    for row_ndx in range(0, data.features.shape[0]):
        if data.output[row_ndx] == prediction(perceptron, data.features[row_ndx, :]):
            num_correct += 1

    return 1.0 - float(num_correct / data.features.shape[0])

