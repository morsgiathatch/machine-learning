import Node
import Metrics
import copy


class Id3:
    def __init__(self):

        self.max_height = 0

    def id3(self, examples, attributes, prev_value, labels, current_depth, max_depth, metric):
        if current_depth > self.max_height:
            self.max_height = current_depth

        if current_depth == max_depth:
            label = get_most_common_label(examples)
            return Node.Node(None, prev_value, label)

        same_label = 1
        base_label = examples[0].get_label()
        for example in examples:
            if example.get_label() != base_label:
                same_label = 0
                break

        if same_label == 1:
            return Node.Node(None, prev_value, base_label)

        if len(attributes) == 0:
            label = get_most_common_label(examples)
            return Node.Node(None, prev_value, label)

        attribute_to_split_on = Metrics.get_splitting_attribute(examples, attributes, labels, metric)

        # Make root node
        node = Node.Node(attribute_to_split_on, prev_value, None)

        # Construct S_v
        for attribute_value in attribute_to_split_on.values:
            examples_less_split_attribute = copy.deepcopy(examples)
            for example in examples:
                if example.get_attribute_value(attribute_to_split_on) != attribute_value:
                    examples_less_split_attribute.remove(example)

            # If S_v is empty, add leaf node containing most common label of S
            if len(examples_less_split_attribute) == 0:
                node.add_child(Node.Node(None, attribute_value, get_most_common_label(examples)))

            else:
                less_attributes = list(copy.deepcopy(attributes))
                less_attributes.remove(attribute_to_split_on)
                node.add_child(self.id3(examples_less_split_attribute, less_attributes, attribute_value, labels,
                               current_depth + 1, max_depth, metric))

        return node

    def get_max_height(self):
        return self.max_height

    def reset_max_height(self):
        self.max_height = 0


# Helper to get the most common label
def get_most_common_label(examples):
    scores = {}
    for example in examples:
        scores[example.get_label()] = 0

    for example in examples:
        scores[example.get_label()] += 1

    label = None
    max_count = 0
    for key in scores:
        if scores[key] > max_count:
            max_count = scores[key]
            label = key

    return label
