from DecisionTree import Problem1
from DecisionTree import Problem2

problem = int(input("\nPlease choose a problem\n1. Problem 1\n2. Problem 2\n"))
valid_choice = True
if problem != 1 and problem != 2:
    valid_choice = False

while not valid_choice:
    print("Incorrect Choice")
    problem = int(input("Please choose a problem\n1. Problem 1\n2. Problem 2\n"))
    if problem == 1 or problem == 2:
        valid_choice = True

if problem == 1:
    Problem1.problem1()
else:
    Problem2.problem2()
