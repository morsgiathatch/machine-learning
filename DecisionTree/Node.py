class Node:
    """Node class for use in decision tree

    """
    # splitting_attribute is the value of the branch leading to this node
    # value is the value of the branch leading to this node from the previous splitting attribute
    # label is the label

    def __init__(self, splitting_attribute, value, label):
        self.splitting_attribute = splitting_attribute
        self.value = value
        self.label = label
        self.child_nodes = []

    def add_child(self, node):
        self.child_nodes.append(node)

    def get_splitting_attribute(self):
        return self.splitting_attribute

    def set_splitting_attribute(self, attribute):
        self.splitting_attribute = attribute

    def get_value(self):
        return self.value

    def get_children(self):
        return self.child_nodes

    def get_label(self):
        return self.label
