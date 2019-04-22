from NeuralNetworks import GradientDescent
from NeuralNetworks import ThreeLayerNN
import sys


def hw5p3():
    redo_problem = True

    while redo_problem:
        problem = int(input("\nPlease choose a problem\n.2 Problem b\n3. Problem c\n4. Problem d\n5. Exit\n"))
        valid_choice = True
        if problem != 2 and problem != 3 and problem != 4 and problem != 5:
            valid_choice = False

        while not valid_choice:
            print("Incorrect Choice")
            problem = int(input("\nPlease choose a problem\n.2 Problem b\n3. Problem c\n4. Problem d\n5. Exit\n"))
            if problem == 2 or problem == 3 or problem == 4 or problem == 5:
                valid_choice = True

        if problem == 2:
            hw5p2()
        elif problem == 3:
            hw5p3()
        elif problem == 4:
            hw5p4()
        else:
            break

        should_redo = str(input("\nWould you like to do another problem from HW5? y/n\n"))
        if should_redo == "n":
            redo_problem = False


def hw5p2():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = BankNoteData.BankNoteData(dir_path + '/../Data/bank_note/train.csv', shift_origin=True)



    

    return None


def hw5p3():
    return None

def hw5p4():
    return None