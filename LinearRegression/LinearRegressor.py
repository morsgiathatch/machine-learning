from numpy import linalg as la


def objective_function(features, output, w_vector):
    return 0.5 * la.norm(features.dot(w_vector) - output)


def obj_gradient_function(features, labels, args):
    weights = args[0]
    return (features.transpose()).dot((features.dot(weights)) - labels)


def stoch_gradient_function(feature, label, args):
    weights = args[0]
    return (label - weights.dot(feature)) * feature


def analytic_solution(features, output):
    return (la.inv((features.transpose()).dot(features))).dot((features.transpose()).dot(output))
