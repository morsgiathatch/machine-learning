import numpy as np


class ThreeLayerNN:
    def __init__(self, num_units_per_layer, weights):   # if we have w_{ij}^k, call weights[i][j][k] to get it
        self.num_units_per_layer = num_units_per_layer
        self.weights = weights
        self.layer0 = []
        self.layer1 = []
        self.layer2 = []
        self.output_vertex = Vertex(0.0, 3, 0)

        # Construct layer 1, 2
        for i in range(0, self.num_units_per_layer):
            self.layer1.append(Vertex(0.0, 1, i))
            self.layer2.append(Vertex(0.0, 2, i))

        for i in range(1, self.num_units_per_layer):
            for j in range(0, self.num_units_per_layer):
                self.layer2[i].add_edge(self.layer1[j], next_=False)

        # edges between layer 2 and 3
        for i in range(0, self.num_units_per_layer):
            self.layer2[i].add_edge(self.output_vertex, next_=True)

    # Call this to predict a label for example x
    def predict(self, x):
        cache = np.array(np.shape(3, self.num_units_per_layer))
        cache.fill(-1.0)
        # Update zeroth layer
        self.layer0 = []
        for i in range(0, self.num_units_per_layer):
            self.layer0.append(Vertex(x[i], 0, i))

        for i in range(1, self.num_units_per_layer):
            for j in range(0, self.num_units_per_layer):
                self.layer1[i].add_edge(self.layer0[j], next_=False)

        val = self.predict_(self.output_vertex, cache=cache)
        return val

    def predict_(self, vertex, cache):
        if vertex.layer == 0:
            return vertex.val

        if vertex.layer == 3:
            _sum = 0.0
            for vert_ in vertex.adj_prev:
                _sum += self.weights[vert_.index, 1, 3] * self.predict_(vert_, cache)
            vertex.val = _sum
            return vertex.val

        if len(vertex.adj_prev) == 0:
            vertex.val = 1.0
            return vertex.val

        _sum = 0.0
        for vert_ in vertex.adj_prev:
            if cache[vert_.layer][vert_.index] == -1.0:
                cache[vert_.layer][vert_.index] = self.predict_(vert_, cache)

            _sum += self.weights[vert_.index, vertex.index, vertex.layer] * cache[vert_.layer][vert_.index]

        vertex.val = sigmoid(_sum)
        return vertex.val

    def update_weights(self, weights):
        self.weights = weights

    def gradient(self, x, y_actual, weights):
        self.update_weights(weights)
        # Update zeroth layer
        self.layer0 = []
        for i in range(0, self.num_units_per_layer):
            self.layer0.append(Vertex(x[i], 0, i))

        for i in range(1, self.num_units_per_layer):
            for j in range(0, self.num_units_per_layer):
                self.layer1[i].add_edge(self.layer0[j], next_=False)

        # Begin backtracking
        y = self.predict(x)
        grad_cache = np.array(np.shape(self.num_units_per_layer, self.num_units_per_layer, 3))
        grad_cache.fill(0.0)

        # Find 3rd layer of derivatives
        for i in range(0, self.num_units_per_layer):
            grad_cache[i, 1, 3] = (y - y_actual) * self.layer2[i].val

        # Find 2nd layer of derivatives
        for i in range(0, self.num_units_per_layer):
            for j in range(1, self.num_units_per_layer):
                grad_cache[i, j, 2] = grad_cache[j, 1, 3] * self.weights[j][1][3] * (1.0 - self.layer2[j].val) * self.layer1[i].val

        # Find 3rd layer of derivatives
        for i in range(0, self.num_units_per_layer):
            for j in range(1, self.num_units_per_layer):
                grad_cache[i, j, 1] = self.layer0[i].val * (1.0 - self.layer1[j].val) * np.sum(np.multiply(self.weights[j, :, 2], grad_cache[j, :, 2]))

        return grad_cache


class Vertex:
    def __init__(self, val, layer, index):
        self.val = val
        self.layer = layer
        self.index = index
        self.adj_next = []
        self.adj_prev = []

    def add_edge(self, vertex, next_):
        if next_ and vertex not in self.adj_next:
            self.adj_next.append(vertex)
            vertex.add_edge(self, not next_)
        elif not next_ and vertex not in self.adj_prev:
            self.adj_prev.append(vertex)
            vertex.add_edge(self, not next_)

    def __eq__(self, other):
        if self.layer != other.layer:
            return False
        if self.index != other.index:
            return False
        return False


def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-1.0 * val))
