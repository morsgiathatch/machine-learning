import numpy as np
import math
import metricModule


# INFORMATION GAIN METRIC
# Definition of Entropy
def entropy(examples, labels):

    # First get proportion values from examples
    p_values = {}
    for label in labels:
        p_values[label] = 0.0

    for example in examples:
        p_values[example.get_label()] += 1.0

    # Calculate entropy

    return metricModule.entropy(list(p_values.values()))
    # _sum = 0.0
    # for p in p_values:
    #     if p_values[p] > 0.0:
    #         pr = float(p_values[p]) / float(len(examples))
    #         _sum += pr * np.log2(pr)
    #
    # _sum *= -1.0
    # return _sum


# Get gain using information gain
def information_gain(examples, attribute, labels):

    gain = get_gain(examples, attribute, labels, entropy)
    return gain


# INFORMATION GAIN METRIC
# Definition of Entropy
def weighted_entropy(examples, labels):

    # First get proportion values from examples
    p_values = {}
    for label in labels:
        p_values[label] = 0.0

    examples_sum = 0.0
    for example in examples:
        p_values[example.get_label()] += example.get_weight()
        examples_sum += example.get_weight()

    # Calculate entropy
    _sum = 0.0
    for p in p_values:
        if p_values[p] > 0.0:
            pr = float(p_values[p]) / examples_sum
            _sum += pr * np.log2(pr)

    _sum *= -1.0
    return _sum


# Get gain using weighted information gain
def weighted_information_gain(examples, attribute, labels):

    gain = get_weighted_gain(examples, attribute, labels, weighted_entropy)
    return gain


# MAJORITY ERROR METRIC
# Definition of Majority Error
def majority_error(examples, labels):
    if len(examples) == 0:
        return 0.0

    # First get distribution of labels
    count = {}
    for label in labels:
        count[label] = 0

    for example in examples:
        count[example.get_label()] += 1

    # Calculate the majority error
    max_occur = 0
    for key in count:
        if count[key] > max_occur:
            max_occur = count[key]

    maj_error = 1.0 - float(max_occur)/float(len(examples))
    return maj_error


# Get gain using majority error
def majority_error_gain(examples, attribute, labels):
    gain = get_gain(examples, attribute, labels, majority_error)
    return gain


# GINI INDEX METRIC
# Helper method to compute Gini Index
def gini_index(examples, labels):
    if len(examples) == 0:
        return 0.0

    # First get proportion values from examples
    p_values = {}
    for label in labels:
        p_values[label] = 0

    for example in examples:
        p_values[example.get_label()] += 1

    sum = 0.0
    for p_value in p_values:
        sum += (float(p_values[p_value] * p_values[p_value]) / float((len(examples) * len(examples))))
    return 1.0 - sum


# Get gain using gini index
def gini_index_gain(examples, attribute, labels):
    gain = get_gain(examples, attribute, labels, gini_index)
    return gain


# Helper method to avoid code duplication
def get_gain(examples, attribute, labels, metric):
    gain = metric(examples, labels)

    for value in attribute.values:
        # Make copy of examples without the attribute value
        examples_v = []
        for example in examples:
            if example.get_attribute_value(attribute) == value:
                examples_v.append(example)

        weighted_gain = float(len(examples_v)) / float(len(examples)) * metric(examples_v, labels)
        gain -= weighted_gain

    return math.fabs(gain)


# Helper method to avoid code duplication
def get_weighted_gain(examples, attribute, labels, metric):
    gain = metric(examples, labels)

    for value in attribute.values:
        # Make copy of examples without the attribute value
        examples_v = []
        examples_v_sum = 0.0
        examples_sum = 0.0
        for example in examples:
            examples_sum += example.get_weight()
            if example.get_attribute_value(attribute) == value:
                examples_v.append(example)
                examples_v_sum += example.get_weight()

        weighted_gain = examples_v_sum / examples_sum * metric(examples_v, labels)
        gain -= weighted_gain

    return math.fabs(gain)


# Helper method to avoid code duplication. Get splitting attribute
def get_splitting_attribute(examples, attributes, labels, metric):
    attribute_to_split_on = None
    gain = 0.0
    for attribute in attributes:
        temp_gain = metric(examples, attribute, labels)
        if temp_gain >= gain:
            gain = temp_gain
            attribute_to_split_on = attribute

    return attribute_to_split_on



