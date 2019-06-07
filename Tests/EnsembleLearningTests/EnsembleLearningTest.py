from Tests.EnsembleLearningTests import Problem2c, Problem2b, Problem2a, Problem2e, Problem2d, CrossTestEnsembleLearning


def ensemble_learning_test():
    redo_problem = True

    while redo_problem:

        problem = int(input("\nPlease choose a test\n1. Problem 2a\n2. Problem 2b\n3. "
                            "Problem 2c\n4. Problem 2d\n5. Problem 2e\n6. Cross examine Ensemble Learning Algorithms on"
                            "different dataset\n7. Exit\n"))

        valid_choice = True
        if problem not in range(1, 8):
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a test\n1. Problem 2a\n2. Problem 2b\n3. "
                                "Problem 2c\n4. Problem 2d\n5. Problem 2e\n6. Cross examine Ensemble Learning "
                                "Algorithms on different dataset\n7. Exit\n"))
            if problem in range(1, 8):
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
        elif problem == 6:
            CrossTestEnsembleLearning.run_cross_comparison()
        else:
            break

        should_redo = str(input("\nWould you like to do another test of ensemble learning algorithms? y/n\n"))
        if should_redo == "n":
            redo_problem = False

