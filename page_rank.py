import numpy as np


def read_txt_data(filename):
    """
    @param filename - str

    prints data from file
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


def get_probability_matrix(graph, weight_of_random_choice):
    vertices_amount = len(graph.keys())
    probability_matrix = [[0] * vertices_amount] * vertices_amount

    for i, i_vertix_id in enumerate(graph.keys()):
        for j, j_vertix_id in enumerate(graph.keys()):
            # if node has outgoing links
            if graph[i_vertix_id] != []:
                probability_matrix[i][j] = 1 / len(graph[i_vertix_id])
            else:
                probability_matrix[i][j] = 1 / vertices_amount

    probability_matrix = np.matrix(probability_matrix)
    probability_matrix = probability_matrix * weight_of_random_choice +\
                         (1 - weight_of_random_choice) *\
                         np.matrix([[1 / vertices_amount] *
                                    vertices_amount] * vertices_amount)

    return probability_matrix
