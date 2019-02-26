from DecisionTree import Problem1
from DecisionTree import Problem2


def hw1():

    redo_problem = True

    while redo_problem:
        problem = int(input("\nPlease choose a problem\n1. Problem 1\n2. Problem 2\n3. Exit\n"))
        valid_choice = True
        if problem != 1 and problem != 2 and problem != 3:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a problem\n1. Problem 1\n2. Problem 2\n3. Exit\n"))
            if problem == 1 or problem == 2 or problem == 3:
                valid_choice = True

        if problem == 1:
            Problem1.problem1()
        elif problem == 2:
            Problem2.problem2()
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW1? y/n\n"))
        if should_redo == "n":
            redo_problem = False
