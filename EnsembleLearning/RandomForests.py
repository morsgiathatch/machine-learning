from DecisionTree import Id3
import random
import numpy as np
import sys


class RandomForests:
    """
    RandomForests class for binary labeled features (-1, 1)
    """
    def __init__(self, features, labels, attributes, t_value, size):
        """
        RandomForests constructor
        :param features: ordered features from dataset
        :type features: python list containing Feature objects
        :param attributes: attributes for current fit iteration
        :type attributes: python tuple containing Attribute objects
        :param t_value: number of decision trees in forest
        :type t_value: integer
        :param size: size of random attribute subset in ID3 iteration
        :type size: integer
        """
        self.t_value = t_value
        self.features = features
        self.attributes = attributes
        self.labels = labels
        self.size = size
        self.forest = []

    def fit(self, print_status_bar):
        """
        train random forests
        :param print_status_bar: set to True if a status printout is desired
        :type print_status_bar: boolean
        :return: None
        :rtype: None
        """
        counter = 1
        toolbar_width = 100
        factor = int(self.t_value / toolbar_width)
        factor = max(factor, 1)
        if print_status_bar:
            print("Building Bagging Trees")
            sys.stdout.write("Progress: [%s]" % (" " * toolbar_width))
            sys.stdout.flush()

        for i in range(0, self.t_value):
            # Get bootstrap example
            bootstrap_sample = get_bootstrap_sample(self.features)
            id3 = Id3.Id3(metric='information_gain')
            id3.fit(features=bootstrap_sample, attributes=self.attributes, prev_value=None, label_set=(-1, 1),
                    current_depth=0, max_depth=float("inf"), rand_attribute_size=self.size)
            self.forest.append(id3)

            if i % factor == 0 and print_status_bar:
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
        Get prediction from single example
        :param example: example with which to make prediction
        :type example: Feature object
        :return: +/- 1.0 label for example
        :rtype: float
        """
        _sum = 0.0
        for tree in self.forest:
            _sum += tree.predict(example)

        _sum /= float(len(self.forest))

        return np.sign(_sum)


def get_bootstrap_sample(examples):
    bootstrap_samples = []
    for j in range(0, int(len(examples))):
        bootstrap_samples.append(examples[random.randint(0, len(examples) - 1)])
    return bootstrap_samples
