def read_txt_data(filename):
    """
    @param filename - str

    prints data from file
    """
    graph = {}
    with open(filename) as graph_file:
        for line in graph_file.readlines():
            line_list = line.strip().split()
            print("line lst: ", line_list)
            if len(line_list) == 1:
                graph[line_list[0]] = []
            else:
                x = line_list[0]
                y = line_list[1:]
                print("x : ", x)
                print("y : ", y)
                if x in graph:
                    graph[x].append(y)
                else:
                    graph[x] = y

        return graph

my_graph = read_txt_data("data_s.txt")
for key in my_graph.keys():
    print("vertix: {}, links to: {}".format(key, my_graph[key]))
