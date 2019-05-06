import os
from Perceptron import BankNoteData, KernelPerceptron


def hw4xc():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/train.csv', shift_origin=True)
    test_data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/test.csv', shift_origin=True)

    gammas = [0.01, 0.1, 0.5, 1., 2., 5., 10., 100.]
    for gamma in gammas:
        print("Running Kernel Perceptron with Gamma = %.16f" % gamma)
        kernel_perceptron = KernelPerceptron.KernelPerceptron(examples=data.features, labels=data.output, gamma=gamma)
        perceptron = kernel_perceptron.run_kernel_perceptron(num_epochs=10)

        train_percentage = get_percentages(data, kernel_perceptron)
        test_percentage = get_percentages(test_data, kernel_perceptron)

        print("Weight vector was:")
        print(perceptron)
        print("Train error percentage was %.16f" % train_percentage)
        print("Test error percentage was %.16f" % test_percentage)


def get_percentages(data, kernel_perceptron):
    num_correct = 0
    for row_ndx in range(0, data.features.shape[0]):
        if data.output[row_ndx] == kernel_perceptron.get_prediction(data.features[row_ndx, :]):
            num_correct += 1

    return 1.0 - float(num_correct / data.features.shape[0])
