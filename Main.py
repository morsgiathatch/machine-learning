from Tests.DecisionTreeTests import DecisionTreeTest
from Tests.EnsembleLearningTests import EnsembleLearningTest
from Tests.LinearRegressionTests import LinearRegressionTest
from Tests.PerceptronTests import PerceptronTest
from Tests.SVMTests import SVMTest


if __name__ == '__main__':
    redo_hm = True

    # This test suite was originally developed for the grading portion of CS5350 at the University of Utah. Some changes
    # were made to reduce the explicit relation to the homeworks.

    while redo_hm:
        print("\n================================================================")
        print("||     Welcome to Corbin Baldwin's Machine Learning Tests     ||")
        print("================================================================")

        test_choice = int(input("Please select a test to run\n1. Decision Tree Tests\n2. Ensemble Learning Tests\n3. "
                                "Linear Regression Tests\n4. Perceptron Tests\n5. SVM Tests\n6. Exit\n"))

        valid_choice = True
        if test_choice not in range(1, 7):
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            test_choice = int(input("Please select a test to run\n1. Decision Tree Tests\n2. Ensemble Learning Tests\n"
                                    "3. Linear Regression Tests\n4. Perceptron Tests\n5. SVM Tests\n6. Exit\n"))
            if test_choice in range(1, 7):
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
            # choice = int(input("Which problem do you wish to view? `2`, `3`, `0` for exit?\n"))
            # hw5_choice = True
            # if choice != 2 and choice != 3 and choice != 0:
            #     hw5_choice = False
            #
            # while not hw5_choice:
            #     print("Incorrect Choice")
            #     choice = int(input("Which problem do you wish to view? `2`, `3`, `0` for exit?\n"))
            #     if choice == 2 or choice == 3 or choice == 0:
            #         hw5_choice = True
            #
            # if choice == 2:
            #     HW5p2.hw5p2()
            # elif choice == 3:
            #     HW5p3.hw5p3()
        else:
            break

        should_redo = str(input("Would you like to run another test? y/n\n"))
        if should_redo == "n":
            redo_hm = False
else:
    exit(-1)
