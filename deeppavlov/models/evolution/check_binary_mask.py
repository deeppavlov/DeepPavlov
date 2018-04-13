import numpy as np
import networkx as nx
import copy


def number_to_type_layer(node_id, n_types):
    # return node_layer, node_type
    return node_id // n_types, node_id % n_types


def type_layer_to_number(node_layer, node_type, n_types):
    return node_layer * n_types + node_type


def find_sources_and_sinks(directed_graph):
    sources = []
    sinks = []

    for i in directed_graph.nodes():
        if directed_graph.in_degree(i) == 0 and directed_graph.out_degree(i) > 0:
            sources.append(i)
        if directed_graph.in_degree(i) > 0 and directed_graph.out_degree(i) == 0:
            sinks.append(i)

    return sources, sinks


def get_digraph_from_binary_mask(nodes, binary_mask):
    directed_graph = nx.DiGraph()
    total_nodes = len(nodes)

    for i in range(total_nodes):
        directed_graph.add_node(i)

    for i in range(total_nodes):
        for j in range(total_nodes):
            if binary_mask[i, j] == 1:
                directed_graph.add_edge(i, j)
    return directed_graph


def get_binary_mask_from_digraph(nodes, directed_graph):
    binary_mask = np.zeros((len(nodes), len(nodes)))
    for edge in directed_graph.edges():
        binary_mask[edge[0], edge[1]] = 1
    return binary_mask
#
#
# def check_binary_mask(nodes, binary_mask):
#     directed_graph = get_digraph_from_binary_mask(nodes, binary_mask)
#     sources, sinks = find_sources_and_sinks(directed_graph)
#
#     while not nx.is_directed_acyclic_graph(directed_graph):
#         cycles = list(nx.simple_cycles(directed_graph))
#         print("Cycles: {}".format(cycles))
#         for cycle_ in cycles:
#             cycle = copy.deepcopy(cycle_) + [cycle_[0]]
#             for i in range(len(cycle_)):
#                 new_directed_graph = copy.deepcopy(directed_graph)
#                 new_directed_graph.remove_edge(cycle[i], cycle[i+1])
#                 new_sources, new_sinks = find_sources_and_sinks(new_directed_graph)
#                 if nx.is_directed_acyclic_graph(new_directed_graph):
#                     if set(new_sources) == set(sources) and set(new_sinks) == set(sinks):
#                         directed_graph.remove_edge(cycle[i], cycle[i+1])
#                         continue
#     binary_mask = get_binary_mask_from_digraph(nodes, directed_graph)
#     return True, binary_mask


def check_binary_mask(nodes, binary_mask):
    directed_graph = get_digraph_from_binary_mask(nodes, binary_mask)
    sources, sinks = find_sources_and_sinks(directed_graph)

    while not nx.is_directed_acyclic_graph(directed_graph):
        candidates = []
        cycles = list(nx.simple_cycles(directed_graph))
        print("Cycles: {}".format(cycles))
        # number of candidates to be the best new graph
        cycles_len = np.array([len(cycle) for cycle in cycles])
        n_candidates = np.prod(cycles_len)

        for i in range(n_candidates):
            new_directed_graph = copy.deepcopy(directed_graph)
            candidates.append(new_directed_graph)

        for j, cycle_ in enumerate(cycles):
            cycle = copy.deepcopy(cycle_) + [cycle_[0]]
            for i in range(len(cycle_)):
                candidates[].remove_edge(cycle[i], cycle[i + 1])
                new_sources, new_sinks = find_sources_and_sinks(new_directed_graph)
                if set(new_sources).issuperset(set(sources)) and set(new_sinks).issuperset(set(sinks)):
                    directed_graph.remove_edge(cycle[i], cycle[i + 1])
                    continue
                else:
                    new_directed_graph.add_edge(cycle[i], cycle[i + 1])

    binary_mask = get_binary_mask_from_digraph(nodes, directed_graph)
    return True, binary_mask
