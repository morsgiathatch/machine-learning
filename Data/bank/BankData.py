class Feature:

    def __init__(self, terms, weight, base_size):
        self.label = terms.pop()
        self.attributes = terms
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
            self.attributes[index] = "yes"
        else:
            self.attributes[index] = "no"

    def convert_to_numeric(self):
        for i in range(0, 16):
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
    """Data class for bank data

    """
    # All class attributes below are hard-coded due to poor data-desc.txt
    age = Attribute((0, 1), 0)
    job = Attribute((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 1)
    marital = Attribute((0, 1, 2), 2)
    education = Attribute((0, 1, 2, 3), 3)
    default = Attribute((0, 1), 4)
    balance = Attribute((0, 1), 5)
    housing = Attribute((0, 1), 6)
    loan = Attribute((0, 1), 7)
    contact = Attribute((0, 1, 2), 8)
    day = Attribute((0, 1), 9)
    month = Attribute((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 10)
    duration = Attribute((0, 1), 11)
    campaign = Attribute((0, 1), 12)
    pdays = Attribute((0, 1), 13)
    previous = Attribute((0, 1), 14)
    poutcome = Attribute((0, 1, 2, 3), 15)

    attributes = (age, job, marital, education, default, balance, housing, loan,
                  contact, day, month, duration, campaign, pdays, previous, poutcome)

    labels = (1, -1)
    labels_map = {"yes": 1, "no": -1}

    age_map = {"yes": 0, "no": 1}
    job_map = {"admin.": 0, "unknown": 1, "unemployed": 2, "management": 3, "housemaid": 4, "entrepreneur": 5,
               "student": 6, "blue-collar": 7, "self-employed": 8, "retired": 9, "technician": 10, "services": 11}
    marital_map = {"married": 0, "divorced": 1, "single": 2}
    education_map = {"unknown": 0, "secondary": 1, "primary": 2, "tertiary": 3}
    default_map = {"yes": 0, "no": 1}
    balance_map = {"yes": 0, "no": 1}
    housing_map = {"yes": 0, "no": 1}
    loan_map = {"yes": 0, "no": 1}
    contact_map = {"unknown": 0, "telephone": 1, "cellular": 2}
    day_map = {"yes": 0, "no": 1}
    month_map = {"jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
                 "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11}
    duration_map = {"yes": 0, "no": 1}
    campaign_map = {"yes": 0, "no": 1}
    pdays_map = {"yes": 0, "no": 1}
    previous_map = {"yes": 0, "no": 1}
    poutcome_map = {"unknown": 0, "other": 1, "failure": 2, "success": 3}

    data_map = [age_map, job_map, marital_map, education_map, default_map, balance_map, housing_map, loan_map,
                contact_map, day_map, month_map, duration_map, campaign_map, pdays_map, previous_map, poutcome_map]

    def __init__(self):
        """Data constructor

        """
        self.examples = []

    def initialize_data_from_file(self, filepath, unknown_is_not_attribute):
        """Initialize data from csv file

        :param filepath: absolute path to csv file
        :type filepath: string
        :param unknown_is_not_attribute: are 'unknown' values considered attributes?
        :type unknown_is_not_attribute: bool
        :return: None
        """
        # Initialize necessary data structures to modify data
        ages = []           # index 0
        balances = []       # index 5
        days = []           # index 9
        durations = []      # index 11
        campaigns = []      # index 12
        pdays = []          # index 13
        previous = []       # index 14
        lists = [ages, balances, days, durations, campaigns, pdays, previous]

        job_distro = {"admin.": 0, "unknown": 0, "unemployed": 0, "management": 0, "housemaid": 0, "entrepreneur": 0,
                      "student": 0, "blue-collar": 0, "self-employed": 0, "retired": 0, "technician": 0, "services": 0}
        education_distro = {"unknown": 0, "secondary": 0, "primary": 0, "tertiary": 0}
        contact_distro = {"unknown": 0, "telephone": 0, "cellular": 0}
        poutcome_distro = {"unknown": 0, "other": 0, "failure": 0, "success": 0}

        with open(filepath, 'r') as f:
            for line in f:
                terms = line.strip().split(',')
                self.examples.append(Feature(terms, 1.0, 0.0))

                ages.append(float(terms[0]))
                balances.append(float(terms[5]))
                days.append(float(terms[9]))
                durations.append(float(terms[11]))
                campaigns.append(float(terms[12]))
                pdays.append(float(terms[13]))
                previous.append(float(terms[14]))

                if unknown_is_not_attribute:
                    job_distro[terms[1]] += 1
                    education_distro[terms[3]] += 1
                    contact_distro[terms[8]] += 1
                    poutcome_distro[terms[15]] += 1

        common_elements = [get_common_element(job_distro), get_common_element(education_distro),
                           get_common_element(contact_distro), get_common_element(poutcome_distro)]

        thresholds = []
        for i in range(0, 7):
            thresholds.append(get_median(sorted(lists[i])))

        for example in self.examples:
            example.set_attribute_value(thresholds[0], 0)
            example.set_attribute_value(thresholds[1], 5)
            example.set_attribute_value(thresholds[2], 9)
            example.set_attribute_value(thresholds[3], 11)
            example.set_attribute_value(thresholds[4], 12)
            example.set_attribute_value(thresholds[5], 13)
            example.set_attribute_value(thresholds[6], 14)
            example.set_weight(1.0)
            example.set_base_size(len(self.examples))

            if unknown_is_not_attribute:
                if example.get_attribute_at(1) == "unknown":
                    example.set_unknown_attribute(common_elements[0], 1)
                if example.get_attribute_at(3) == "unknown":
                    example.set_unknown_attribute(common_elements[1], 3)
                if example.get_attribute_at(8) == "unknown":
                    example.set_unknown_attribute(common_elements[2], 8)
                if example.get_attribute_at(15) == "unknown":
                    example.set_unknown_attribute(common_elements[3], 15)

        for example in self.examples:
            example.convert_to_numeric()


# Get double Median of list
def get_median(values):
    length = len(values)
    if length % 2 == 0:
        return float(values[int(length / 2)] + values[int(length / 2) - 1]) / 2.0
    else:
        return float(values[length / 2])


# Get key with most amount values in dictionary
def get_common_element(values):
    common_key = None
    count = 0
    for key in values:
        if values[key] > count:
            count = values[key]
            common_key = key

    return common_key
