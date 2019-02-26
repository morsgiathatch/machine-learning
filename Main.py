from DecisionTree import HW1
from EnsembleLearning import HW2p2
from LinearRegression import HW2p4

redo_hm = True

while redo_hm:
    print("Welcome to Corbin Baldwin's Programming Assignments")
    homework = int(input("Please select a homework to choose from\n1. HW1\n2. HW2\n3. Exit\n"))

    valid_choice = True
    if homework != 1 and homework != 2 and homework != 3:
        valid_choice = False

    while not valid_choice:
        print("Incorrect Choice")
        homework = int(input("Please select a homework to choose from\n1. HW1\n2. HW2\n3. Exit\n"))
        if homework == 1 or homework == 2 or homework == 3:
            valid_choice = True

    if homework == 1:
        HW1.hw1()
    elif homework == 2:
        choice = int(input("Which problem do you wish to view? `2`, `4` or `0` for exit?\n"))
        valid_choice = True
        if choice != 2 and choice != 4 and choice != 0:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            choice = int(input("Which problem do you wish to view? `2` or `4`?\n"))
            if choice == 2 or choice == 4 or choice == 0:
                valid_choice = True

        if choice == 2:
            HW2p2.hw2p2()
        elif choice == 4:
            HW2p4.hw2p4()

    else:
        break

    should_redo = str(input("Would you like to review another homework? y/n\n"))
    if should_redo == "n":
        redo_hm = False
