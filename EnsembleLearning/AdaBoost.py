from DecisionTree import Id3
import numpy as np
import sys


class Adaboost:
    def __init__(self, features, attributes, labelset, t_value):
        self.features = features
        # Construct features and labels
        for feature in self.features:
            feature.set_weight(float(1/len(self.features)))
        self.labels = []
        for feature in self.features:
            self.labels.append(feature.get_label())
        self.labels = np.array(self.labels)

        self.attributes = attributes
        self.labelset = labelset
        self.t_value = t_value
        self.alphas = np.zeros(self.t_value)
        self.h_classifiers = [None] * self.t_value
        self.dt = np.ones(len(self.features)) / float(len(self.features))
        self.h_predictions = np.zeros(len(self.features))

    def run_adaboost(self, print_status=False):
        if print_status:
            print("Building AdaBoost trees")
            sys.stdout.write("Progress: 0 / %s" % self.t_value)
            sys.stdout.flush()

        for i in range(0, self.t_value):
            id3 = Id3.Id3(metric='weighted_information_gain')
            id3.run_id3(features=self.features, attributes=self.attributes, prev_value=None, labels=self.labelset,
                        current_depth=0, max_depth=1)
            self.h_classifiers[i] = id3
            # Get predictions
            for i_, feature in enumerate(self.features):
                self.h_predictions[i_] = float(id3.get_prediction(feature))

            epsilon = self.get_epsilon()
            self.alphas[i] = 0.5 * np.log((1.0 - epsilon) / epsilon)
            self.update_dt(self.alphas[i])

            # Update weights
            for j, feature in enumerate(self.features):
                feature.set_weight(self.dt[j])

            if print_status:
                sys.stdout.write("\rProgress: %s / %s" % (i + 1, self.t_value))
                sys.stdout.flush()

        if print_status:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def get_epsilon(self):
        return 0.5 - (0.5 * self.dt.dot(np.multiply(self.labels, self.h_predictions)))

    def update_dt(self, alpha):
        self.dt *= np.exp(-alpha * np.multiply(self.labels, self.h_predictions))
        self.dt /= np.sum(self.dt)

    def get_prediction(self, example):
        sum_ = 0.0
        for i in range(0, self.t_value):
            sum_ += self.alphas[i] * self.h_classifiers[i].get_prediction(example)

        return np.sign(sum_)
