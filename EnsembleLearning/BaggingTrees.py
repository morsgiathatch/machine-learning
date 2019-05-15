from DecisionTree import Id3
import numpy as np
import random
import sys


class BaggingTrees:
    def __init__(self, t_value, features, attributes, labels, attribute_factor):
        self.t_value = t_value
        self.features = features
        self.attributes = attributes
        self.labels = labels
        self.attribute_factor = attribute_factor
        self.trees = []

    def run_bagging_trees(self, print_status_bar):
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
            id3.run_id3(features=sample_features, attributes=self.attributes, prev_value=None,
                        labels=self.labels, current_depth=0, max_depth=float("inf"))
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

    def get_prediction(self, example):
        _sum = 0.0
        for i in range(0, self.t_value):
            _sum += self.trees[i].get_prediction(example)

        _sum /= float(len(self.trees))

        return np.sign(_sum)


def get_sample(examples, factor):
    samples = []
    for j in range(0, int(len(examples) / factor)):
        samples.append(examples[random.randint(0, len(examples) - 1)])
    return samples
