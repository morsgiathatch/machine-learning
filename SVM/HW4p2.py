def hw4p2():
    redo_problem = True

    while redo_problem:

        problem = int(input("\nPlease choose a problem part\n1. "
                            "Problem 2a\n2. Problem 2b\n3. Problem 2c\n4. Exit\n"))

        valid_choice = True
        if problem != 1 and problem != 2 and problem != 3 and problem != 4:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a problem part4\n1. "
                                "Problem 2a\n2. Problem 2b\n3. Problem 2c\n4. Exit\n"))

            if problem == 1 or problem == 2 or problem == 3 or problem == 4:
                valid_choice = True

        if problem == 1:
            choice_a()
        elif problem == 2:
            choice_b()
        elif problem == 3:
            choice_c()
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW 4 problem 2? y/n\n"))
        if should_redo == "n":
            redo_problem = False


def choice_a():
    return None


def choice_b():
    return None


def choice_c():
    return None
