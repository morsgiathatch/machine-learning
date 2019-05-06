from Tests.DecisionTreeTests import DecisionTreeTest
from EnsembleLearning import HW2p2
from EnsembleLearning import ExtraCredit
from LinearRegression import HW2p4
from Perceptron import HW3p2
from SVM import HW4
from LogisticRegression import HW5p2
from NeuralNetworks import HW5p3

redo_hm = True

while redo_hm:
    print("\n================================================================")
    print("||     Welcome to Corbin Baldwin's Machine Learning Tests     ||")
    print("================================================================")

    test_choice = int(input("Please select a test to run\n1. Decision Tree Tests\n2. HW2\n3. HW3\n4. HW4\n5. HW5\n6. Exit\n"))

    valid_choice = True
    if test_choice != 1 and test_choice != 2 and test_choice != 3 and test_choice != 4 and test_choice != 5 and test_choice != 6:
        valid_choice = False

    while not valid_choice:
        print("Incorrect Choice")
        test_choice = int(input("Please select a test to run\n1. Decision Tree Tests\n2. HW2\n3. HW3\n4. HW4\n5. HW5\n6. Exit\n"))
        if test_choice == 1 or test_choice == 2 or test_choice == 3 or test_choice == 4 or test_choice == 5 or test_choice == 6:
            valid_choice = True

    if test_choice == 1:
        DecisionTreeTest.decision_tree_test()
    elif test_choice == 2:
        choice = int(input("Which problem do you wish to view? `2`, `3`, `4` or `0` for exit?\n"))
        valid_choice = True
        if choice != 2 and choice != 3 and choice != 4 and choice != 0:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            choice = int(input("Which problem do you wish to view? `2` or `4`?\n"))
            if choice == 2 or choice == 3 or choice == 4 or choice == 0:
                valid_choice = True

        if choice == 2:
            HW2p2.hw2p2()
        elif choice == 3:
            ExtraCredit.extra_credit()
        elif choice == 4:
            HW2p4.hw2p4()
    elif test_choice == 3:
        HW3p2.hw3p2()
    elif test_choice == 4:
        HW4.hw4()
    elif test_choice == 5:
        choice = int(input("Which problem do you wish to view? `2`, `3`, `0` for exit?\n"))
        hw5_choice = True
        if choice != 2 and choice != 3 and choice != 0:
            hw5_choice = False

        while not hw5_choice:
            print("Incorrect Choice")
            choice = int(input("Which problem do you wish to view? `2`, `3`, `0` for exit?\n"))
            if choice == 2 or choice == 3 or choice == 0:
                hw5_choice = True

        if choice == 2:
            HW5p2.hw5p2()
        elif choice == 3:
            HW5p3.hw5p3()
    else:
        break

    should_redo = str(input("Would you like to review another test? y/n\n"))
    if should_redo == "n":
        redo_hm = False
