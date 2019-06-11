from Tests.EnsembleLearningTests import adaboost_bagged_cross_comparison, bagged_trees_test, adaboost_test, forest_bagged_cross_comparison, random_forest_test, CrossTestEnsembleLearning


def ensemble_learning_test():
    redo_problem = True

    while redo_problem:

        problem = int(input("\nPlease choose a test\n1. Adaboost Test\n2. Bagged Trees Test\n3. "
                            "Adaboost and Bagged Trees Cross Comparison\n4. Random Forest Test\n"
                            "5. Random Forest and Bagged Trees Cross Comparison\n"
                            "6. Cross examine Ensemble Learning Algorithms on Different Dataset\n7. Exit\n"))

        valid_choice = True
        if problem not in range(1, 8):
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a test\n1. Adaboost Test\n2. Bagged Trees Test\n3. "
                                "Adaboost and Bagged Trees Cross Comparison\n4. Random Forest Test\n"
                                "5. Random Forest and Bagged Trees Cross Comparison\n"
                                "6. Cross examine Ensemble Learning Algorithms on Different Dataset\n7. Exit\n"))
            if problem in range(1, 8):
                valid_choice = True

        if problem == 1:
            adaboost_test.adaboost_test()
        elif problem == 2:
            bagged_trees_test.bagged_trees_test()
        elif problem == 3:
            adaboost_bagged_cross_comparison.adaboost_bagged_cross_comparison()
        elif problem == 4:
            random_forest_test.random_forest_test()
        elif problem == 5:
            forest_bagged_cross_comparison.forest_bagged_cross_comparison()
        elif problem == 6:
            CrossTestEnsembleLearning.run_cross_comparison()
        else:
            break

        should_redo = str(input("\nWould you like to do another test of ensemble learning algorithms? y/n\n"))
        if should_redo == "n":
            redo_problem = False

