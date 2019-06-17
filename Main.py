from Tests.DecisionTreeTests import DecisionTreeTest
from Tests.EnsembleLearningTests import EnsembleLearningTest
from Tests.LinearRegressionTests import LinearRegressionTest
from Tests.PerceptronTests import PerceptronTest
from Tests.SVMTests import SVMTest
from Tests.LogisticRegressionTests import LogisticRegressionTest
from Tests.NeuralNetworkTests import NeuralNetworkTest


if __name__ == '__main__':
    redo_hm = True

    # This test suite was originally developed for the grading portion of CS5350 at the University of Utah. Some changes
    # were made to reduce the explicit relation to the homeworks.

    while redo_hm:
        print("\n================================================================")
        print("||     Welcome to Corbin Baldwin's Machine Learning Tests     ||")
        print("================================================================")

        test_choice = int(input("Please select a test to run\n1. Decision Tree Tests\n2. Ensemble Learning Tests\n3. "
                                "Linear Regression Tests\n4. Perceptron Tests\n5. SVM Tests\n"
                                "6. Logistic Regression Tests\n7. Neural Network Tests\n8. Exit\n"))

        valid_choice = True
        if test_choice not in range(1, 9):
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            test_choice = int(input("Please select a test to run\n1. Decision Tree Tests\n2. Ensemble Learning Tests\n3. "
                                    "Linear Regression Tests\n4. Perceptron Tests\n5. SVM Tests\n"
                                    "6. Logistic Regression Tests\n7. Neural Network Tests\n8. Exit\n"))
            if test_choice in range(1, 9):
                valid_choice = True

        if test_choice == 1:
            DecisionTreeTest.decision_tree_test()
        elif test_choice == 2:
            EnsembleLearningTest.ensemble_learning_test()
        elif test_choice == 3:
            LinearRegressionTest.linear_regression_test()
        elif test_choice == 4:
            PerceptronTest.perceptron_test()
        elif test_choice == 5:
            SVMTest.svm_test()
        elif test_choice == 6:
            LogisticRegressionTest.logistic_regression_test()
        elif test_choice == 7:
            NeuralNetworkTest.neural_network_test()
        else:
            break

        should_redo = str(input("Would you like to run another test? y/n\n"))
        if should_redo == "n":
            redo_hm = False
else:
    exit(-1)
