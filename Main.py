from DecisionTree import HW1
from EnsembleLearning import HW2p2
from EnsembleLearning import ExtraCredit
from LinearRegression import HW2p4
from Perceptron import HW3p2
from SVM import HW4p2

redo_hm = True

while redo_hm:
    print("\n================================================================")
    print("||    Welcome to Corbin Baldwin's Programming Assignments     ||")
    print("================================================================")

    homework = int(input("Please select a homework to choose from\n1. HW1\n2. HW2\n3. HW3\n4. HW4\n5. Exit\n"))

    valid_choice = True
    if homework != 1 and homework != 2 and homework != 3 and homework != 4 and homework != 5:
        valid_choice = False

    while not valid_choice:
        print("Incorrect Choice")
        homework = int(input("Please select a homework to choose from\n1. HW1\n2. HW2\n3. HW3\n4. HW4\n5. Exit\n"))
        if homework == 1 or homework == 2 or homework == 3 or homework == 4 or homework == 5:
            valid_choice = True

    if homework == 1:
        HW1.hw1()
    elif homework == 2:
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
    elif homework == 3:
        HW3p2.hw3p2()
    elif homework == 4:
        HW4p2.hw4p2()
    else:
        break

    should_redo = str(input("Would you like to review another homework? y/n\n"))
    if should_redo == "n":
        redo_hm = False
