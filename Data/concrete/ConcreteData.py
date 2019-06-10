import numpy as np


class ConcreteData:
    def __init__(self, filepath):
        self.features = []
        self.output = []

        # Due to time limitations, exceptions are not handled
        with open(filepath, 'r') as f:
            for line in f:
                terms = line.strip().split(',')
                self.output.append(float(terms.pop()))
                temp_list = []
                temp_list.append(1.0)  # Add 1.0 for first term to get bias
                for term in terms:
                    temp_list.append(float(term.rstrip('\n')))
                self.features.append(temp_list)

        self.features = np.array(self.features)
        self.output = np.array(self.output)
