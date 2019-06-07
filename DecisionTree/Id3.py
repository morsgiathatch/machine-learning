from DecisionTree import Metrics
from DecisionTree import Node
import random
import copy


class Id3:
    """Id3 class to store state.

    """
    metrics = {'information_gain': Metrics.information_gain, 'majority_error_gain': Metrics.majority_error_gain,
               'gini_index_gain': Metrics.gini_index_gain, 'weighted_information_gain': Metrics.weighted_information_gain}

    def __init__(self, metric):
        """ Constructor for ID3.

        :param metric: type of information metric to use
        :type metric: string
        """
        self.max_height = 0
        self.root = None
        self.metric = Id3.metrics[metric]

    def fit(self, features, attributes, prev_value, labels, current_depth, max_depth, rand_attribute_size=None):
        """train Id3 decision tree

        :param features: ordered features from dataset
        :type features: python list containing Feature objects
        :param attributes: attributes for current fit iteration
        :type attributes: python tuple containing Attribute objects
        :param prev_value: attribute value of previous adjacent node
        :type prev_value: integer or None
        :param labels: ordered labels from dataset
        :type labels: python tuple containing possible integer labels
        :param current_depth: current tree depth
        :type current_depth: integer
        :param max_depth: maximum desired tree depth
        :type max_depth: integer or float
        :param rand_attribute_size: size of desired random attribute subset if not None
        :type rand_attribute_size: integer or None
        :return: root node of decision tree
        :rtype: Node.Node
        """
        if current_depth > self.max_height:
            self.max_height = current_depth

        if current_depth == max_depth:
            label = get_most_common_label(features)
            return Node.Node(None, prev_value, label)

        same_label = 1
        base_label = features[0].get_label()
        for example in features:
            if example.get_label() != base_label:
                same_label = 0
                break

        if same_label == 1:
            return Node.Node(None, prev_value, base_label)

        if len(attributes) == 0:
            label = get_most_common_label(features)
            return Node.Node(None, prev_value, label)

        if rand_attribute_size is not None:
            indices = random.sample(range(0, len(attributes)), min(rand_attribute_size, len(attributes)))
            random_attributes = []
            for index in indices:
                random_attributes.append(attributes[index])

            attribute_to_split_on = Metrics.get_splitting_attribute(features, random_attributes, labels, self.metric)
        else:
            attribute_to_split_on = Metrics.get_splitting_attribute(features, attributes, labels, self.metric)

        # Make root node
        node = Node.Node(attribute_to_split_on, prev_value, None)

        # Construct S_v
        for attribute_value in attribute_to_split_on.values:
            examples_less_split_attribute = []
            for example in features:
                if example.get_attribute_value(attribute_to_split_on) == attribute_value:
                    examples_less_split_attribute.append(example)

            # If S_v is empty, add leaf node containing most common label of S
            if len(examples_less_split_attribute) == 0:
                node.add_child(Node.Node(None, attribute_value, get_most_common_label(features)))

            else:
                less_attributes = list(copy.deepcopy(attributes))
                less_attributes.remove(attribute_to_split_on)
                node.add_child(self.fit(examples_less_split_attribute, less_attributes, attribute_value, labels,
                                        current_depth + 1, max_depth, rand_attribute_size))

        if prev_value is None:
            self.root = node
        else:
            return node

    def get_max_height(self):
        return self.max_height

    def reset_max_height(self):
        self.max_height = 0

    def predict(self, feature):
        """Predict label for example

        :param feature: feature to be used for prediction
        :type feature: Feature
        :return: prediction
        :rtype: integer
        """
        return self.get_prediction_helper(feature, self.root)

    # Helper method
    def get_prediction_helper(self, feature, node):
        if node.get_splitting_attribute() is None:
            return node.get_label()

        next_node = None
        attribute = node.get_splitting_attribute()
        for i in range(len(node.get_children())):
            value = node.child_nodes[i].get_value()
            if value == feature.get_attribute_value(attribute):
                next_node = node.child_nodes[i]

        return self.get_prediction_helper(feature, next_node)


# Helper to get the most common label
def get_most_common_label(examples):
    scores = {}
    for example in examples:
        scores[example.get_label()] = 0.0

    for example in examples:
        scores[example.get_label()] += example.get_weight()

    label = None
    max_count = 0.0
    for key in scores:
        if scores[key] > max_count:
            max_count = scores[key]
            label = key

    return label


# Unordered attribute list comparison. Essentially set equality
def attr_equal(attributes1, attributes2):
    if len(attributes1) != len(attributes2):
        return False

    for item1 in list(attributes1):
        if item1 not in list(attributes2):
            return False

    for item2 in list(attributes2):
        if item2 not in list(attributes1):
            return False

    return True
