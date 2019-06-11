from DecisionTree import Id3
import numpy as np
import random
import sys


class BaggingTrees:
    """
    BaggingTrees class for binary labeled data (-1, 1)
    """
    def __init__(self, features, attributes, t_value, attribute_factor):
        """
        BaggingTrees constructor
        :param features: ordered features from dataset
        :type features: python list containing Feature objects
        :param attributes: attributes for current fit iteration
        :type attributes: python tuple containing Attribute objects
        :param t_value: Number of decision trees
        :type t_value: integer
        :param attribute_factor: factor of features to sample. I.e. 5 is a 1/5th of samples
        :type attribute_factor: float
        """
        self.t_value = t_value
        self.features = features
        self.attributes = attributes
        self.attribute_factor = attribute_factor
        self.trees = []

    def fit(self, print_status_bar):
        """
        train bagging trees
        :param print_status_bar: set to True if a status printout is desired
        :type print_status_bar: boolean
        :return: None
        :rtype: None
        """
        counter = 1
        toolbar_width = 100
        fractor = int(self.t_value / toolbar_width)
        fractor = max(fractor, 1)
        if print_status_bar:
            print("Building Bagging Trees")
            sys.stdout.write("Progress: [%s]" % (" " * toolbar_width))
            sys.stdout.flush()
        for i in range(0, self.t_value):
            sample_features = get_sample(self.features, self.attribute_factor)

            # Run Id3 and keep root node
            id3 = Id3.Id3(metric='information_gain')
            id3.fit(features=sample_features, attributes=self.attributes, prev_value=None,
                    label_set=(-1, 1), current_depth=0, max_depth=float("inf"))
            self.trees.append(id3)

            if i % fractor == 0 and print_status_bar:
                sys.stdout.write('\r')
                sys.stdout.flush()
                sys.stdout.write('Progress: [%s' % ('#' * counter))
                sys.stdout.write('%s]' % (' ' * (toolbar_width - counter)))
                sys.stdout.flush()
                counter += 1

        if print_status_bar:
            print("")

    def predict(self, example):
        """
        get prediction from single example
        :param example: example with which to make prediction
        :type example: Feature object
        :return: +/- 1.0 label for example
        :rtype: float
        """
        _sum = 0.0
        for i in range(0, self.t_value):
            _sum += self.trees[i].predict(example)

        _sum /= float(len(self.trees))

        return np.sign(_sum)


def get_sample(examples, factor):
    samples = []
    for j in range(0, int(len(examples) / factor)):
        samples.append(examples[random.randint(0, len(examples) - 1)])
    return samples
