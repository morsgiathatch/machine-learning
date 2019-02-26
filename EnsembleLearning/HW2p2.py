from EnsembleLearning import Problem2a
from EnsembleLearning import Problem2b
from EnsembleLearning import Problem2c
from EnsembleLearning import Problem2d
from EnsembleLearning import Problem2e


def hw2p2():
    redo_problem = True

    while redo_problem:

        problem = int(input("\nPlease choose a problem\n1. Problem 2a\n2. Problem 2b\n3. "
                            "Problem 2c\n4. Problem 2d\n5. Problem 2e\n6. Exit\n"))

        valid_choice = True
        if problem != 1 and problem != 2 and problem != 3 and problem != 4 and problem != 5 and problem != 6:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a problem\n1. Problem 2a\n2. Problem 2b\n3. "
                                "Problem 2c\n4. Problem 2d\n5. Problem 2e\n6. Exit\n"))
            if problem == 1 or problem == 2 or problem == 3 or problem == 4 or problem == 5 or problem == 6:
                valid_choice = True

        if problem == 1:
            Problem2a.problem2a()
        elif problem == 2:
            Problem2b.problem2b()
        elif problem == 3:
            Problem2c.problem2c()
        elif problem == 4:
            Problem2d.problem2d()
        elif problem == 5:
            Problem2e.problem2e()
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW2p2? y/n\n"))
        if should_redo == "n":
            redo_problem = False

