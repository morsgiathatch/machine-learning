from Tests.LinearRegressionTests import run_linear_regression_test


def linear_regression_test():
    redo_test = True

    while redo_test:

        test_choice = int(input("\nPlease choose a test below\n1. "
                                "Gradient Descent\n2. Stochastic Gradient Descent\n3. Exact Solution\n4. Exit\n"))

        valid_choice = True
        if test_choice not in range(1, 5):
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            test_choice = int(input("\nPlease choose a test below\n1. "
                                    "Gradient Descent\n2. Stochastic Gradient Descent\n3. Exact Solution\n4. Exit\n"))

            if test_choice in range(1, 5):
                valid_choice = True

        if test_choice != 4:
            run_linear_regression_test.run_linear_regression_test(test_choice)
        else:
            break

        should_redo = str(input("\nWould you like to run another linear regression test? y/n\n"))
        if should_redo == "n":
            redo_test = False


