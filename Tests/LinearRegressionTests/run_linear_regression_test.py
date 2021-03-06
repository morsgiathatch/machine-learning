from Data.concrete import ConcreteData
from Algorithms import GradientDescent
from LinearRegression import LinearRegressor as lr
import os
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


def run_linear_regression_test(part):
    # Construct data sets

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = ConcreteData.ConcreteData(dir_path + '/../../Data/concrete/train.csv')

    # Test data
    test_data = ConcreteData.ConcreteData(dir_path + '/../../Data/concrete/test.csv')

    if part != 3:

        manual_input = str(input("Do you wish to input parameters manually? y/n\n"))
        analytic_answer = str(input("Do you wish to print out the analytic answer for comparison? y/n\n"))
        if manual_input == "y":
            max_iters = int(input("Please enter max number of iterations for gradient descent\n"))
            tolerance = float(input("Please enter a tolerance for algorithm termination\n"))
            step_size = float(input("Please enter a constant step size for each iteration\n"))
        else:
            max_iters = 10000
            tolerance = float(1e-6)
            step_size = 0.02

        grad_desc = GradientDescent.GradientDescent(data.features, data.output)
        if part == 1:
            step_size = 0.0145
            result = grad_desc.fit(obj_func=lr.objective_function, grad_func=lr.obj_gradient_function,
                                   args=None, max_iters=max_iters, step_function=lambda x: step_size,
                                   tolerance=tolerance)
        else:
            result = grad_desc.fit_stochastic(obj_func=lr.objective_function,
                                              grad_func=lr.stoch_gradient_function, args=None,
                                              max_iters=max_iters, step_function=lambda x: step_size,
                                              tolerance=tolerance)

        print("step size was " + str(step_size))
        if result[1] == max_iters + 1:
            print("Iteration did not converge after " + str(max_iters) + " iterations. Resultant vector is:")
            print(result[0])
            print("Try with another constant step size.")
        else:
            print("Success! Converged after " + str(result[1]) + " iterations. Resultant vector is: ")
            print(result[0])

        print("\nEvaluating at the vector yields train cost of %.16f"
              % (lr.objective_function(data.features, data.output, [result[0]])))
        t_vals = np.linspace(0, (len(result[2])) * 10, len(result[2]))

        print("Using the above, we then have the following test cost:")
        test_cost = lr.objective_function(test_data.features, test_data.output, [result[0]])
        print("%.16f" % test_cost)

        if analytic_answer == "y":
            print("\nAnalytic minimizer is:")
            minimizer = lr.analytic_solution(data.features, data.output)
            print(minimizer)
            print("Cost using analytic minimizer is: %.16f"
                  % (lr.objective_function(data.features, data.output, [minimizer])))
            minimizer_cost = lr.objective_function(test_data.features, test_data.output, [minimizer])
            print("Test cost using analytic minimizer is: %.16f" % minimizer_cost)
            print("\n2-norm error in result vector vs. minimizer is %.16f" % (la.norm(result[0] - minimizer)))

            print("Absolute error in test cost vs analytic test cost is %.16f" % (np.abs(minimizer_cost - test_cost)))
            print("Relative error in test cost vs analytic test cost is %.16f"
                  % (np.abs((minimizer_cost - test_cost) / minimizer_cost)))

        plt.plot(t_vals, result[2], label='Cost')
        plt.legend(loc='best')
        plt.show()

    else:
        print("\nAnalytic minimizer is:")
        minimizer = lr.analytic_solution(data.features, data.output)
        print(minimizer)
        print("Cost using analytic minimizer is: %.16f"
              % (lr.objective_function(data.features, data.output, [minimizer])))
        print("Test cost using analytic minimizer is: %.16f"
              % (lr.objective_function(test_data.features, test_data.output, [minimizer])))
