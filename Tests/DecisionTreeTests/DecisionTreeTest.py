from Tests.DecisionTreeTests import non_numeric_id3_test, numeric_id3_test


def decision_tree_test():
    redo_problem = True

    while redo_problem:
        test_choice = int(input("\nPlease make a selection\n1. Test non-numeric ID3\n2. Test numeric ID3\n3. Exit\n"))
        valid_choice = True
        if test_choice != 1 and test_choice != 2 and test_choice != 3:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            test_choice = int(input("\nPlease make a selection\n1. Test non-numeric ID3\n2. Test numeric ID3\n3. Exit\n"))
            if test_choice == 1 or test_choice == 2 or test_choice == 3:
                valid_choice = True

        if test_choice == 1:
            non_numeric_id3_test.non_numeric_id3_test()
        elif test_choice == 2:
            numeric_id3_test.numeric_id3_test()
        else:
            break

        should_redo = str(input("\nWould you like to do another decision tree test? y/n\n"))
        if should_redo == "n":
            redo_problem = False
