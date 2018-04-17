import numpy as np
import networkx as nx
from copy import copy, deepcopy


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


def check_and_correct_binary_mask(nodes, binary_mask_):
    binary_mask = deepcopy(binary_mask_)
    binary_mask = np.array(binary_mask)
    directed_graph = get_digraph_from_binary_mask(nodes, binary_mask)
    sources, sinks = find_sources_and_sinks(directed_graph)

    while not nx.is_directed_acyclic_graph(directed_graph):
        candidates = []
        cycles = list(nx.simple_cycles(directed_graph))
        n_cycles = len(cycles)
        # print("Cycles: {}".format(cycles))
        # number of candidates to be the best new graph
        cycles_len = np.array([len(cycle) for cycle in cycles])
        n_candidates = np.prod(cycles_len)

        for i in range(n_candidates):
            new_directed_graph = copy.deepcopy(directed_graph)
            for j in range(n_cycles):
                node_id = (i // np.prod(cycles_len[:j])) % cycles_len[j]
                new_directed_graph.remove_edge(cycles[j][node_id], cycles[j][(node_id + 1) % cycles_len[j]])
            candidates.append(new_directed_graph)

        best_cand = None
        best_diff = 10e10
        for i in range(n_candidates):
            new_sources, new_sinks = find_sources_and_sinks(candidates[i])
            if set(new_sources) == set(sources) and set(new_sinks) == set(sinks):
                best_cand = candidates[i]
            elif (len(set(new_sources).difference(set(sources))) +
                  len(set(new_sinks).difference(set(sinks))) < best_diff):
                best_cand = candidates[i]
                best_diff = len(set(new_sources).difference(set(sources))) + len(set(new_sinks).difference(set(sinks)))

        directed_graph = best_cand

    binary_mask = get_binary_mask_from_digraph(nodes, directed_graph)
    return binary_mask

# def get_graph_and_plot(nodes, binary_mask, n_types):
#     import matplotlib.pyplot as plt
#
#     total_nodes = len(nodes)
#     dg = nx.DiGraph()
#
#     for i in range(total_nodes):
#         dg.add_node(i)
#
#     pos = {}
#
#     for i in range(total_nodes):
#         for j in range(total_nodes):
#             if binary_mask[i,j] == 1:
#                 dg.add_edge(i, j)
#         pos[i] = np.array(number_to_type_layer(i, n_types))[::-1]
#
#     plt.figure(figsize=(6, 6))
#     nx.draw(dg, pos, node_color='b', node_size=5000, alpha=0.3)
#
#     nx.draw_networkx_labels(dg, pos, nodes, font_size=18)
#     plt.show()
