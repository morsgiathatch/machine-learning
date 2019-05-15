from DecisionTree import Id3
import random
import numpy as np
import sys


class RandomForests:
    def __init__(self, t_value, features, attributes, labels, size):
        self.t_value = t_value
        self.features = features
        self.attributes = attributes
        self.labels = labels
        self.size = size
        self.forest = []

    def run_random_forests(self, print_status_bar):

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
            id3.run_id3(features=bootstrap_sample, attributes=self.attributes, prev_value=None, labels=self.labels,
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

    def get_prediction(self, example):
        _sum = 0.0
        for tree in self.forest:
            _sum += tree.get_prediction(example)

        _sum /= float(len(self.forest))

        return np.sign(_sum)


def get_bootstrap_sample(examples):
    bootstrap_samples = []
    for j in range(0, int(len(examples))):
        bootstrap_samples.append(examples[random.randint(0, len(examples) - 1)])
    return bootstrap_samples
