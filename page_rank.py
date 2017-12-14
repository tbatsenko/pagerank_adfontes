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
    probability_matrix = [[0 for i1 in range(vertices_amount)] for i2 in range(vertices_amount)]

    for i, i_vertix_id in enumerate(graph.keys()):
        for j, j_vertix_id in enumerate(graph.keys()):
            # if node has outgoing links
            if graph[i_vertix_id] != []:
                if j_vertix_id in graph[i_vertix_id]:
                    probability_matrix[i][j] = 1 / len(graph[i_vertix_id])
                else:
                    probability_matrix[i][j] = 0

    # Adjust with weight_of_random_walk constant
    probability_matrix = np.matrix(probability_matrix)
    main_part = probability_matrix * (1 - weight_of_random_walk)
    probability_vector = np.matrix([[1 / vertices_amount] * vertices_amount] * vertices_amount)
    random_walk_part = weight_of_random_walk * probability_vector
    probability_matrix = main_part + random_walk_part
    return probability_matrix


def pagerank(probability_matrix, previous_pr):
    transition_matrix = probability_matrix.transpose()
    epsilon = 10 ** -3
    # Make delta greater than epsilon for first iteration
    delta = epsilon + 1

    while delta > epsilon:
        current_pr = transition_matrix.dot(previous_pr)
        w = np.linalg.norm(previous_pr, ord=1) - np.linalg.norm(current_pr, ord=1)
        current_pr = current_pr + w * (1 / len(previous_pr))

        delta = np.linalg.norm(current_pr - previous_pr, ord=1)
        previous_pr = current_pr

    return current_pr


def adaptive_pagerank(probability_matrix, previous_pr):
    transition_matrix = probability_matrix.transpose()
    initial_transition_matrix = deepcopy(transition_matrix)
    # Set up accuracy
    epsilon = 10 ** -3
    # Make delta greater than epsilon for first iteration
    delta = epsilon + 1
    n = transition_matrix.shape[0]
    final_pr = []
    converged_pr = previous_pr

    while delta > epsilon:
        current_pr = transition_matrix.dot(previous_pr)
        w = np.linalg.norm(previous_pr, ord=1) - np.linalg.norm(current_pr, ord=1)
        current_pr = current_pr + w * (1 / len(previous_pr))

        for i in range(transition_matrix.shape[1]):
            if abs(current_pr[i] - previous_pr[i]) / previous_pr[i] < epsilon:
                transition_matrix[i] = np.zeros(transition_matrix.shape[1])
                # pr_vector_to_save_indx[i] = 0
            else:
                converged_pr[i] = 0

        delta = np.linalg.norm(current_pr - previous_pr, ord=1)
        previous_pr = current_pr

    # Fix structure of final_pr
    return current_pr

def launch():
    small_sample = 'data/sample-small.txt'
    large_sample = 'data/sample-large.txt'
    chosen_sample = small_sample
    weight_of_random_walk = 0.15
    probability_matrix = get_probability_matrix(get_graph_from_txt_data(chosen_sample), weight_of_random_walk)
    initial_pr = np.full((len(probability_matrix), 1), 1 / len(probability_matrix))
    probability_matrix = np.matrix(probability_matrix)

    result = adaptive_pagerank(probability_matrix, initial_pr)
    result.sort()
    print(len(result))
    print(sum(result))
    print(result)

launch()
