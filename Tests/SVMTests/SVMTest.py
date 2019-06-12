from Tests.SVMTests import PrimalSVMTest, DualSVMTest


def svm_test():

    redo_tests = True

    while redo_tests:
        test_choice = int(input("\nPlease choose a selection of tests\n1. Primal SVM\n2. Dual SVM\n3. Exit\n"))
        valid_choice = True
        if test_choice not in range(1, 4):
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            test_choice = int(input("\nPlease choose a selection of tests\n1. Primal SVM\n2. Dual SVM\n3. Exit\n"))
            if test_choice in range(1, 4):
                valid_choice = True

        if test_choice == 1:
            PrimalSVMTest.primal_svm_test()
        elif test_choice == 2:
            DualSVMTest.dual_svm_test()
        else:
            break

        should_redo = str(input("\nWould you like to run another SVM test? y/n\n"))
        if should_redo == "n":
            redo_tests = False