import collections

import networkx as nx


def load_graph(path):
    # Use a nested dictionary to store the graph as an adjacency structure:
    # G[u][v] = weight
    G = collections.defaultdict(dict)

    # Read the edge list file line by line.
    with open(path) as text:
        for line in text:
            vertices = line.strip().split()

            # Read two endpoint nodes and their edge weight.
            # If the dataset does not provide weights, the default weight is set to 1.0.
            # (This implementation currently assumes an unweighted edge list and uses 1.0.)
            v_i, v_j, weight = int(vertices[0]), int(vertices[1]), 1.0

            # Since the graph is undirected, store the edge in both directions.
            G[v_i][v_j] = weight
            G[v_j][v_i] = weight

    return G


class my_Graph:
    def __init__(self):
        # Initialize an undirected NetworkX graph object.
        self.graph = nx.Graph()

    def createGraph(self, filename):
        # Read an edge list file and construct a NetworkX graph.
        file = open(filename, 'r')
        for line in file.readlines():
            nodes = line.split()

            # Parse the two endpoint nodes of the edge.
            source = int(nodes[0])
            target = int(nodes[1])

            # Use default edge weight = 1.0 for unweighted datasets.
            weight = 1.0

            # Add the edge to the graph (weight is stored as an edge attribute).
            self.graph.add_edge(int(source), int(target), weight=int(weight))

        return self.graph