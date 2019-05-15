class Feature:

    def __init__(self, terms):
        self.label = terms.pop()
        self.attributes = terms
        for i in range(0, 6):
            self.attributes[i] = Data.data_map[i][self.attributes[i]]
        self.label = Data.labels_map[self.label]

    def get_attributes(self):
        return self.attributes

    def get_label(self):
        return self.label

    def get_weight(self):
        return 1.0

    def get_attribute_value(self, attribute):
        return self.attributes[attribute.index]

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
    """Data class for car data

    """
    # All class attributes below are hard-coded due to poor data-desc.txt
    buying = Attribute((0, 1, 2, 3), 0)
    maint = Attribute((0, 1, 2, 3), 1)
    doors = Attribute((0, 1, 2, 3), 2)
    persons = Attribute((0, 1, 2), 3)
    lug_boot = Attribute((0, 1, 2), 4)
    safety = Attribute((0, 1, 2), 5)

    attributes = (buying, maint, doors, persons, lug_boot, safety)

    labels = (0, 1, 2, 3)
    labels_map = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}

    buying_map = {"vhigh": 0, "high": 1, "med": 2, "low": 3}
    maint_map = {"vhigh": 0, "high": 1, "med": 2, "low": 3}
    doors_map = {"2": 0, "3": 1, "4": 2, "5more": 3}
    persons_map = {"2": 0, "4": 1, "more": 2}
    lug_boot_map = {"small": 0, "med": 1, "big": 2}
    safety_map = {"low": 0, "med": 1, "high": 2}

    data_map = [buying_map, maint_map, doors_map, persons_map, lug_boot_map, safety_map]

    def __init__(self):
        """Data constructor

        """
        self.examples = []

    def initialize_data_from_file(self, filepath):
        """Initialize data from csv file

        :param filepath: absolute path to csv file
        :type filepath: string
        :return: None
        """
        with open(filepath, 'r') as f:
            for line in f:
                terms = line.strip().split(',')
                self.examples.append(Feature(terms))
