import numpy as np


class ThreeLayerNN:
    def __init__(self, num_units_per_layer, weights):   # if we have w_{ij}^k, call weights[i][j][k] to get it
        self.num_units_per_layer = num_units_per_layer
        self.weights = weights
        self.layer1 = []
        self.layer2 = []
        self.output_vertex = Vertex(0.0, 3, 0)

        # Construct layer 1, 2
        for i in range(0, self.num_units_per_layer):
            self.layer1.append(Vertex(0.0, 1, i))
            self.layer2.append(Vertex(0.0, 2, i))

        for i in range(1, self.num_units_per_layer):
            for j in range(0, self.num_units_per_layer):
                self.layer2[i].add_edge(self.layer1[j], self.weights[j][i][2], next_=False)

        # edges between layer 2 and 3
        for i in range(0, self.num_units_per_layer):
            self.layer2[i].add_edge(self.output_vertex, self.weights[i][1][3], next_=True)

    # Call this to predict a label for example x
    def predict(self, x):
        cache = np.array(np.shape(3, self.num_units_per_layer))
        cache.fill(-1.0)
        layer0 = []
        for i in range(0, self.num_units_per_layer):
            layer0.append(Vertex(x[i], 0, i))

        for i in range(1, self.num_units_per_layer):
            for j in range(0, self.num_units_per_layer):
                self.layer1[i].add_edge(layer0[j], self.weights[j][i][1], next_=False)

        val = self.predict_(self.output_vertex, prev_vertex=None, cache=cache)
        return val

    def predict_(self, vertex, prev_vertex, cache):
        if vertex.layer == 0:
            return vertex.val * vertex.adj_next[prev_vertex]

        if vertex.layer == 3:
            _sum = 0.0
            for vert_ in vertex.adj_prev:
                _sum += vert_.adj_next[vertex] * self.predict_(vert_, vertex, cache)
            return _sum

        if len(vertex.adj_prev) == 0:
            return 1.0

        _sum = 0.0
        for vert_ in vertex.adj_prev:
            if cache[vert_.layer][vert_.index] == -1.0:
                cache[vert_.layer][vert_.index] = self.predict_(vert_, vertex, cache)

            _sum += vert_.adj_next[vertex] * cache[vert_.layer][vert_.index]

        return sigmoid(_sum)

    def update_weights(self, weights):
        self.weights = weights

class Vertex:
    def __init__(self, val, layer, index):
        self.val = val
        self.layer = layer
        self.index = index
        self.vertices = []
        self.adj_next = {}
        self.adj_prev = {}

    def add_edge(self, vertex, weight, next_):
        if vertex not in self.vertices:
            self.vertices.append(vertex)
            if next_:
                self.adj_next[vertex] = weight
            else:
                self.adj_prev[vertex] = weight
            vertex.add_edge(self, weight, not next_)

    def remove_edge(self, vertex):
        if vertex in self.vertices:
            self.vertices.remove(vertex)
            if vertex in self.adj_next:
                del self.adj_next[vertex]
            else:
                del self.adj_prev[vertex]
            vertex.remove_edge(self)

    def __eq__(self, other):
        if self.layer != other.layer:
            return False
        if self.index != other.index:
            return False
        return False


def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-1.0 * val))
