import numpy as np
from copy import deepcopy


def get_graph_from_txt_data(filename):
    """
    @param filename - str

    reads data from file and creates

    @requirements
    The files need to be tab-delimited adjacency list representations
    of the graphs.
    The first token on each line represents the unique id of the source node,
    and the rest of the tokens represent the target nodes
    (i.e., outlinks from the source node).
    If a node does not have any outlinks, its corresponding line will contain
    only one token (the source node id).

    @return dict:
    key    : value
    str(id): [str(id), str(id), ...]
    """
    graph = {}
    with open(filename) as graph_file:
        for line in graph_file.readlines():
            line_list = line.strip().split()
            if len(line_list) == 1:
                graph[line_list[0]] = []
            else:
                x = line_list[0]
                y = line_list[1:]
                if x in graph:
                    graph[x].append(y)
                else:
                    graph[x] = y

        return graph


def get_probability_matrix(graph, weight_of_random_walk):
    """
    Transition matrix describing the transition from i to j
    is given by matrix P with P[i][j] = 1/deg(i)
    where deg(i) - amount of outgoing links from i

    To avoid division by zero in cases where i has no outgoing links,
    we add random walk P[i][j] = 1/n
    where n - amount of vertices in the graph

    Then we add teleportation - possibility to travel from any node in graph
    to any another node. That is implemented by adding weight_of_random_walk.

    :param graph: dict
        key    : value
        str(id): [str(id), str(id), ...]
    :param weight_of_random_walk: float
    :return: np.matrix
    """
    vertices_amount = len(graph.keys())
    P = [[0 for i1 in range(vertices_amount)] for i2 in range(vertices_amount)]
    v = np.matrix([1 / vertices_amount for i in range(vertices_amount)])
    e = np.matrix([1 for i in range(vertices_amount)])
    d = np.matrix([0 if graph[id] else 1 for id in graph.keys()])
    D = d.transpose().dot(v)

    for i, i_vertix_id in enumerate(graph.keys()):
        for j, j_vertix_id in enumerate(graph.keys()):
            # if node has outgoing links
            if graph[i_vertix_id] != []:
                if j_vertix_id in graph[i_vertix_id]:
                    P[i][j] = 1 / len(graph[i_vertix_id])
                else:
                    P[i][j] = 0

    P_prime = np.matrix(P) + D
    E = e.transpose() * v

    probability_matrix = (1 - weight_of_random_walk) * P_prime + \
        weight_of_random_walk * E

    return probability_matrix.transpose()


def adaptive_pagerank(probability_matrix, initial_pr):
    transition_matrix = probability_matrix
    initial_transition_matrix = deepcopy(transition_matrix)
    # Set up accuracy
    epsilon = 10 ** -3
    # Make delta greater than epsilon for first iteration
    delta = epsilon + 1

    previous_pr = initial_pr
    previous_pr_conv = np.matrix(np.zeros(len(previous_pr))).transpose()

    while delta > epsilon:
        # Calculate pagerank for non-converged entities
        current_pr_non_conv = transition_matrix.dot(previous_pr)
        # Calculate pagerank for converged entities
        current_pr_conv = deepcopy(previous_pr_conv)
        # Join converged and non-converged parts
        current_pr = current_pr_non_conv + current_pr_conv

        for i in range(transition_matrix.shape[0]):
            if previous_pr[i] != 0 and abs(current_pr[i] - previous_pr[i]) / previous_pr[i] < epsilon:
                transition_matrix[i] = np.zeros(transition_matrix.shape[1])
                current_pr_conv[i] = deepcopy(current_pr[i])
                current_pr_non_conv[i] = 0

        final_pr = initial_transition_matrix.dot(previous_pr)

        delta = np.linalg.norm(final_pr - previous_pr, ord=1)

        previous_pr = deepcopy(current_pr)
        previous_pr_conv = deepcopy(current_pr_conv)

    return previous_pr


def launch():
    super_small_sample = 'data/sample-super-small.txt'
    small_sample = 'data/sample-small.txt'
    large_sample = 'data/sample-large.txt'
    chosen_sample = small_sample
    weight_of_random_walk = 0.15
    probability_matrix = get_probability_matrix(get_graph_from_txt_data(chosen_sample), weight_of_random_walk)
    initial_pr = np.matrix([1 / probability_matrix.shape[0] for i in range(probability_matrix.shape[0])]).transpose()

    result = adaptive_pagerank(probability_matrix, initial_pr).tolist()
    flat_result = [item for sublist in result for item in sublist]
    flat_result.sort()
    print(flat_result)
    print(len(flat_result))
    print(sum(flat_result))


launch()
