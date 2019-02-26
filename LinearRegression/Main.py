from LinearRegression import Problem4


problem = int(input("\nPlease choose a subproblem of problem4\n1. Problem 4a\n2. Problem 4b\n3. Problem 4c\n"))

valid_choice = True
if problem != 1 and problem != 2 and problem != 3:
    valid_choice = False

while not valid_choice:
    print("Incorrect Choice")
    problem = int(input("\nPlease choose a subproblem of problem4\n1. Problem 4a\n2. Problem 4b\n3. Problem 4c\n"))

    if problem == 1 or problem == 2 or problem == 3:
        valid_choice = True

Problem4.problem4(problem)



