from Tests.LinearRegressionTests import Problem4


def hw2p4():
    redo_problem = True

    while redo_problem:

        problem = int(input("\nPlease choose a problem part4\n1. "
                            "Problem 4a\n2. Problem 4b\n3. Problem 4c\n4. Exit\n"))

        valid_choice = True
        if problem != 1 and problem != 2 and problem != 3 and problem != 4:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a subproblem of problem4\n1. "
                                "Problem 4a\n2. Problem 4b\n3. Problem 4c\n4. Exit"))

            if problem == 1 or problem == 2 or problem == 3 or problem == 4:
                valid_choice = True

        if problem != 4:
            Problem4.problem4(problem)
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW 2 problem 4? y/n\n"))
        if should_redo == "n":
            redo_problem = False

