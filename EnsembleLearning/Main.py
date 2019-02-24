from EnsembleLearning import Problem2a
from EnsembleLearning import Problem2b
from EnsembleLearning import Problem2c
from EnsembleLearning import Problem2d
from EnsembleLearning import Problem2e


problem = int(input("\nPlease choose a problem\n1. Problem 2a\n2. Problem 2b\n3. "
                    "Problem 2c\n4. Problem 2d\n5. Problem 2e\n"))

valid_choice = True
if problem != 1 and problem != 2 and problem != 3 and problem != 4 and problem != 5:
    valid_choice = False

while not valid_choice:
    print("Incorrect Choice")
    problem = int(input("\nPlease choose a problem\n1. Problem 2a\n2. Problem 2b\n3. "
                        "Problem 2c\n4. Problem 2d\n5. Problem 2e\n"))
    if problem == 1 or problem == 2:
        valid_choice = True

if problem == 1:
    Problem2a.problem2a()
elif problem == 2:
    Problem2b.problem2b()
elif problem == 3:
    Problem2c.problem2c()
elif problem == 4:
    Problem2d.problem2d()
else:
    Problem2e.problem2e()


