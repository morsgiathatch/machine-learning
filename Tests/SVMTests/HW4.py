from Tests.SVMTests import HW4XC, HW4p2, HW4p3


def hw4():

    redo_problem = True

    while redo_problem:
        problem = int(input("\nPlease choose a problem\n2. Problem 2\n3. Problem 3\n4. Extra Credit\n5. Exit\n"))
        valid_choice = True
        if problem != 2 and problem != 3 and problem != 4 and problem != 5:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a problem\n2. Problem 2\n3. Problem 3\n4. Extra Credit\n5. Exit\n"))
            if problem == 2 or problem == 3 or problem == 4 or problem == 5:
                valid_choice = True

        if problem == 2:
            HW4p2.hw4p2()
        elif problem == 3:
            HW4p3.hw4p3()
        elif problem == 4:
            HW4XC.hw4xc()
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW4? y/n\n"))
        if should_redo == "n":
            redo_problem = False