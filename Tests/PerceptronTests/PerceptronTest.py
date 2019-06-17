import os
from Data.bank_note import BankNoteData
from Perceptron import Perceptron
import numpy as np
import sys
from Tests.PerceptronTests import KernelPerceptronTest


def perceptron_test():
    redo_tests = True

    while redo_tests:

        test_choice = int(input("\nPlease choose a test below\n1. "
                                "Standard Perceptron\n2. Voted Perceptron\n3. Average Perceptron\n"
                                "4. Cross Comparison of the Above Three\n5. Kernel Perceptron\n6. Exit\n"))

        valid_choice = True
        if test_choice not in range(1, 7):
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            test_choice = int(input("\nPlease choose a test below\n1. "
                                    "Standard Perceptron\n2. Voted Perceptron\n3. Average Perceptron\n"
                                    "4. Cross Comparison of the Above Three\n5. Kernel Perceptron\n6. Exit\n"))

            if test_choice in range(1, 7):
                valid_choice = True

        if test_choice == 1:
            choice_a()
        elif test_choice == 2:
            choice_b()
        elif test_choice == 3:
            choice_c()
        elif test_choice == 4:
            choice_d()
        elif test_choice == 5:
            KernelPerceptronTest.kernel_perceptron_test()
        else:
            break

        should_redo = str(input("\nWould you like to run another perceptron test? y/n\n"))
        if should_redo == "n":
            redo_tests = False


def choice_a():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../../Data/bank_note/train.csv', shift_origin=True)

    test_data = BankNoteData.BankNoteData(dir_path + '/../../Data/bank_note/test.csv', shift_origin=True)

    perceptron = Perceptron.Perceptron(10, data.features, data.output, 0.5)
    perceptron.fit()

    train_percentage = get_percentages(perceptron, data)
    test_percentage = get_percentages(perceptron, test_data)

    print("Weight vector was:")
    print(perceptron.weights)
    print("Train error percentage was %.16f" % train_percentage)
    print("Test error percentage was %.16f" % test_percentage)


def choice_b():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../../Data/bank_note/train.csv', shift_origin=True)

    test_data = BankNoteData.BankNoteData(dir_path + '/../../Data/bank_note/test.csv', shift_origin=True)

    perceptron = Perceptron.Perceptron(10, data.features, data.output, 0.5, perceptron_type='voted')
    perceptron.fit()

    print("The vectors and their respective count are:")
    np.set_printoptions(precision=4)
    for i in range(0, len(perceptron.weights[0])):
        print(perceptron.weights[0][i], "  : %s" % (str(perceptron.weights[1][i])), )

    train_percentage = get_percentages(perceptron, data)
    test_percentage = get_percentages(perceptron, test_data)

    print("Train error percentage was %.16f" % train_percentage)
    print("Test error percentage was %.16f" % test_percentage)


def choice_c():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../../Data/bank_note/train.csv', shift_origin=True)

    test_data = BankNoteData.BankNoteData(dir_path + '/../../Data/bank_note/test.csv', shift_origin=True)

    perceptron = Perceptron.Perceptron(10, data.features, data.output, 0.5, perceptron_type='averaged')
    perceptron.fit()

    train_percentage = get_percentages(perceptron, data)
    test_percentage = get_percentages(perceptron, test_data)

    print("Weight vector was:")
    print(perceptron.weights)
    print("Train error percentage was %.16f" % train_percentage)
    print("Test error percentage was %.16f" % test_percentage)


def choice_d():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../../Data/bank_note/train.csv', shift_origin=True)
    test_data = BankNoteData.BankNoteData(dir_path + '/../../Data/bank_note/test.csv', shift_origin=True)

    train_errors = [[], [], []]
    test_errors = [[], [], []]

    print("Calculating statistics")
    sys.stdout.write('Progress: [%s]' % (' ' * 100))
    sys.stdout.flush()
    for i in range(1, 101):
        standard_perceptron = Perceptron.Perceptron(10, data.features, data.output, 0.5)
        standard_perceptron.fit()
        voted_perceptron = Perceptron.Perceptron(10, data.features, data.output, 0.5, perceptron_type='voted')
        voted_perceptron.fit()
        averaged_perceptron = Perceptron.Perceptron(10, data.features, data.output, 0.5, perceptron_type='averaged')
        averaged_perceptron.fit()

        train_errors[0].append(get_percentages(standard_perceptron, data))
        train_errors[1].append(get_percentages(voted_perceptron, data))
        train_errors[2].append(get_percentages(averaged_perceptron, data))

        test_errors[0].append(get_percentages(standard_perceptron, test_data))
        test_errors[1].append(get_percentages(voted_perceptron, test_data))
        test_errors[2].append(get_percentages(averaged_perceptron, test_data))

        sys.stdout.write('\rProgress: [%s' % ('#' * i))
        sys.stdout.write('%s]' % (' ' * (100 - i)))
        sys.stdout.flush()

    print("\n%24s   %10s   %10s" % ("Standard", "Voted", "Averaged"))
    print("Train Errors: %.8f   %.8f   %.8f" % (float(np.mean(np.array(train_errors[0]))),
                                                float(np.mean(np.array(train_errors[1]))),
                                                float(np.mean(np.array(train_errors[2])))))
    print("Test Errors:  %.8f   %.8f   %.8f" % (float(np.mean(np.array(test_errors[0]))),
                                                float(np.mean(np.array(test_errors[1]))),
                                                float(np.mean(np.array(test_errors[2])))))


def get_percentages(perceptron, data):
    num_correct = 0
    for row_ndx in range(0, data.features.shape[0]):
        if data.output[row_ndx] == perceptron.predict(data.features[row_ndx, :]):
            num_correct += 1

    return 1.0 - float(num_correct / data.features.shape[0])
