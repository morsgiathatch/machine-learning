import numpy as np


class Feature:

    def __init__(self, features, label, weight, base_size):
        self.label = label
        self.attributes = features
        self.weight = weight
        self.base_size = base_size

    def get_attributes(self):
        return self.attributes

    def get_label(self):
        return self.label

    def get_attribute_value(self, attribute):
        return self.attributes[attribute.index]

    def get_attribute_at(self, index):
        return self.attributes[index]

    def get_weight(self):
        return self.weight

    def get_base_size(self):
        return self.base_size

    def set_attribute_value(self, median, index):
        if float(self.attributes[index]) < median:
            self.attributes[index] = "1"
        else:
            self.attributes[index] = "0"

    def convert_to_numeric(self):
        for i in range(0, 23):
            self.attributes[i] = Data.data_map[i][self.attributes[i]]
        self.label = Data.labels_map[self.label]

    def set_unknown_attribute(self, val, index):
        self.attributes[index] = val

    def set_weight(self, weight):
        self.weight = weight

    def set_base_size(self, base_size):
        self.base_size = base_size

    def __eq__(self, other):
        return self.attributes == other.attributes


class Attribute:

    def __init__(self, values, index):
        self.values = values
        self.index = index

    def __eq__(self, other):
        return self.index == other.index


# Define data structures
class Data:
    """Data class for credit default data

    """
    labels = (-1, 1)
    labels_map = {"1": 1, "0": -1}
    attr_0 = Attribute((0, 1), 0)
    attr_1 = Attribute((0, 1, 2), 1)
    attr_2 = Attribute((0, 1, 2, 3, 4, 5, 6), 2)
    attr_3 = Attribute((0, 1, 2, 3), 3)
    attr_4 = Attribute((0, 1), 4)
    attr_5 = Attribute((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 5)
    attr_6 = Attribute((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 6)
    attr_7 = Attribute((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 7)
    attr_8 = Attribute((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 8)
    attr_9 = Attribute((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 9)
    attr_10 = Attribute((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 10)
    attr_11 = Attribute((0, 1), 11)
    attr_12 = Attribute((0, 1), 12)
    attr_13 = Attribute((0, 1), 13)
    attr_14 = Attribute((0, 1), 14)
    attr_15 = Attribute((0, 1), 15)
    attr_16 = Attribute((0, 1), 16)
    attr_17 = Attribute((0, 1), 17)
    attr_18 = Attribute((0, 1), 18)
    attr_19 = Attribute((0, 1), 19)
    attr_20 = Attribute((0, 1), 20)
    attr_21 = Attribute((0, 1), 21)
    attr_22 = Attribute((0, 1), 22)

    attributes = [attr_0, attr_1, attr_2, attr_3, attr_4, attr_5, attr_6, attr_7, attr_8, attr_9, attr_10, attr_11,
                  attr_12, attr_13, attr_14, attr_15, attr_16, attr_18, attr_19, attr_20, attr_21, attr_22]

    attr_0_map = {"1": 0, "0": 1}
    attr_1_map = {"1": 0, "2": 1, "0": 2}
    attr_2_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "0": 5, "6": 6}
    attr_3_map = {"1": 0, "2": 1, "3": 2, "0": 3}
    attr_4_map = {"1": 0, "0": 1}
    attr_5_map = {"-1": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "-2": 10, "0": 11}
    attr_6_map = {"-1": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "-2": 10, "0": 11}
    attr_7_map = {"-1": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "-2": 10, "0": 11}
    attr_8_map = {"-1": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "-2": 10, "0": 11}
    attr_9_map = {"-1": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "-2": 10, "0": 11}
    attr_10_map = {"-1": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "-2": 10, "0": 11}
    attr_11_map = {"1": 0, "0": 1}
    attr_12_map = {"1": 0, "0": 1}
    attr_13_map = {"1": 0, "0": 1}
    attr_14_map = {"1": 0, "0": 1}
    attr_15_map = {"1": 0, "0": 1}
    attr_16_map = {"1": 0, "0": 1}
    attr_17_map = {"1": 0, "0": 1}
    attr_18_map = {"1": 0, "0": 1}
    attr_19_map = {"1": 0, "0": 1}
    attr_20_map = {"1": 0, "0": 1}
    attr_21_map = {"1": 0, "0": 1}
    attr_22_map = {"1": 0, "0": 1}

    data_map = [attr_0_map, attr_1_map, attr_2_map, attr_3_map, attr_4_map, attr_5_map, attr_6_map, attr_7_map,
                attr_8_map, attr_9_map, attr_10_map, attr_11_map, attr_12_map, attr_13_map, attr_14_map, attr_15_map,
                attr_16_map, attr_17_map, attr_18_map, attr_19_map, attr_20_map, attr_21_map, attr_22_map, ]

    def __init__(self):
        """Data constructor

        """
        self.examples = []
        self.data_map = []
        self.attributes = []

    def initialize_data_from_file(self, filepath, start_ndx=0, column_ignore=(), features_to_map_to_num=(), features_to_make_median=(), median_key=None):
        """Initialize data from csv file

        :param filepath: absolute path to csv file
        :type filepath: string
        :param start_ndx: line number to start reading csv. 0-based
        :type start_ndx: integer
        :param column_ignore: indices of columns of csv file to ignore. 0-based.
        :type column_ignore: python tuple of integers
        :return: None
        :rtype: None
        """

        dict_of_vals_to_track = {}

        with open(filepath, 'r') as f:
            for i in range(0, start_ndx):
                f.readline()
            line = f.readline()
            first_line = True
            while line:
                terms = line.strip().split(',')
                # Read each line and ignore unneeded columns, isolate features and label
                features = []
                for i, term in enumerate(terms):
                    if i not in column_ignore:
                        features.append(term)
                label = features.pop()

                # Initialize data map
                if first_line:
                    first_line = False
                    self.data_map = [{}] * len(features)

                # Update data structures
                for i in range(0, len(features)):
                    # Update data map
                    if i in features_to_map_to_num:
                        if features[i] not in self.data_map[i]:
                            self.data_map[i] = len(self.data_map) - 1

                    # Update attributes

                    # Update values to track to set median
                    if i in features_to_make_median:
                        if i not in dict_of_vals_to_track:
                            dict_of_vals_to_track[i] = []
                            dict_of_vals_to_track[i].append(float(features[i]))
                        else:
                            dict_of_vals_to_track[i].append(float(features[i]))



                self.examples.append(Feature(features, label, 1.0, 0.0))

        thresholds = []

        for key in dict_of_vals_to_track:
            thresholds.append(get_median(dict_of_vals_to_track[key]))

        for example in self.examples:
            example.set_attribute_value(thresholds[0], 0)
            example.set_attribute_value(thresholds[1], 4)
            example.set_attribute_value(thresholds[2], 11)
            example.set_attribute_value(thresholds[3], 12)
            example.set_attribute_value(thresholds[4], 13)
            example.set_attribute_value(thresholds[5], 14)
            example.set_attribute_value(thresholds[6], 15)
            example.set_attribute_value(thresholds[7], 16)
            example.set_attribute_value(thresholds[8], 17)
            example.set_attribute_value(thresholds[9], 18)
            example.set_attribute_value(thresholds[10], 19)
            example.set_attribute_value(thresholds[11], 20)
            example.set_attribute_value(thresholds[12], 21)
            example.set_attribute_value(thresholds[13], 22)
            example.set_weight(1.0)

        for example in self.examples:
            example.convert_to_numeric()


    def update_data_structures(self, features, label):


    def get_features(self, subset_type):
        """
        get set of features if necessary
        :param subset_type: type of features. Either 'all', 'train', or 'test'
        :type subset_type: string
        :return: features
        :rtype: numpy array
        """
        features = []
        if subset_type == 'all':
            feature_set = self.examples
        elif subset_type == 'train':
            feature_set = self.train_examples
        elif subset_type == 'test':
            feature_set = self.test_examples
        else:
            raise ValueError('Not valid subset type. See documentation.')
        for example in feature_set:
            features.append(example.get_attributes())

        return np.array(features)

    def get_labels(self, subset_type):
        """
        get set of labels if necessary
        :param subset_type: type of features. Either 'all', 'train', or 'test'
        :type subset_type: string
        :return: labels
        :rtype: numpy array
        """
        labels = []
        if subset_type == 'all':
            feature_set = self.examples
        elif subset_type == 'train':
            feature_set = self.train_examples
        elif subset_type == 'test':
            feature_set = self.test_examples
        else:
            raise ValueError('Not valid subset type. See documentation.')
        for example in feature_set:
            labels.append(example.get_label)

        return np.array(labels)


# Get double Median of list
def get_median(values):
    length = len(values)
    if length % 2 == 0:
        return float(values[int(length / 2)] + values[int(length / 2) - 1]) / 2.0
    else:
        return float(values[length / 2])
