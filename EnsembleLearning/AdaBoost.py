from DecisionTree import Id3
import numpy as np
import sys


class Adaboost:
    """
    Adaboost class for binary labeled data in (-1, 1)
    """
    def __init__(self, features, attributes, t_value):
        """
        Adaboost constructor
        :param features: ordered features from dataset
        :type features: python list containing Feature objects
        :param attributes: attributes for current fit iteration
        :type attributes: python tuple containing Attribute objects
        :param t_value: number of decision stumps
        :type t_value: integer
        """
        self.features = features
        # Construct features and labels
        for feature in self.features:
            feature.set_weight(float(1/len(self.features)))
        self.labels = []
        for feature in self.features:
            self.labels.append(feature.get_label())
        self.labels = np.array(self.labels)

        self.attributes = attributes
        self.labels = (-1, 1)
        self.t_value = t_value
        self.alphas = np.zeros(self.t_value)
        self.h_classifiers = [None] * self.t_value
        self.dt = np.ones(len(self.features)) / float(len(self.features))
        self.h_predictions = np.zeros(len(self.features))

    def fit(self, print_status=False):
        """
        train Adaboost
        :param print_status: set to True if a status printout is desired
        :type print_status: boolean
        :return: None
        :rtype: None
        """
        if print_status:
            print("Building AdaBoost trees")
            sys.stdout.write("Progress: 0 / %s" % self.t_value)
            sys.stdout.flush()

        for i in range(0, self.t_value):
            id3 = Id3.Id3(metric='weighted_information_gain')
            id3.fit(features=self.features, attributes=self.attributes, prev_value=None, labels=self.labels,
                    current_depth=0, max_depth=1)
            self.h_classifiers[i] = id3
            # Get predictions
            for i_, feature in enumerate(self.features):
                self.h_predictions[i_] = float(id3.predict(feature))

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

    def predict(self, example):
        """
        get prediction from single example
        :param example: example with which to make prediction
        :type example: Feature object
        :return: +/- 1.0 label for example
        :rtype: float
        """
        sum_ = 0.0
        for i in range(0, self.t_value):
            sum_ += self.alphas[i] * self.h_classifiers[i].get_prediction(example)

        return np.sign(sum_)
