import numpy as np
import scipy
import copy
from numpy.linalg import matrix_rank

import fpgaconvnet.tools.graphs as graphs

# turn off invalid NaN division
np.seterr(divide='ignore', invalid='ignore')

def get_node_list_matrix(graph):
    return ['input',*list(graph.nodes()),'output']

def get_edge_list_matrix(graph):
    input_nodes  = graphs.get_input_nodes(graph, allow_multiport=True)
    output_nodes = graphs.get_output_nodes(graph, allow_multiport=True)

    edge_list = [(None, input_node) for input_node in input_nodes]
    edge_list += [(node, edge) for node in graph.nodes() for edge in graphs.get_next_nodes(graph, node)]
    edge_list += [(output_node, None) for output_node in output_nodes]

    return edge_list

def get_edges_in(node, edge_list):
    return [edge for edge in edge_list if edge[1] == node]

def get_edges_out(node, edge_list):
    return [edge for edge in edge_list if edge[0] == node]

def matrix_to_graph(matrix,node_list,edge_list):

    # initialise graph
    graph = {}
    edges = []

    # iterate over edges and nodes
    for edge_index in range(matrix.shape[0]):
        node_in  = None
        node_out = None
        for node_index in range(matrix.shape[1]):
            # ignore inputs and outputs
            if node_index == 0 or node_index == len(node_list)-1:
                continue
            # find the input node
            if matrix[edge_index,node_index] > 0.0:
                node_in  = node_list[node_index]
            # find the output node
            if matrix[edge_index,node_index] < 0.0:
                node_out = node_list[node_index]
        # append to edges
        edges.append((node_in,node_out))

    # update graph
    for edge in edges:
        # ignore none connections
        if not edge[0]:
            continue
        # create nodes
        if not edge[0] in graph:
            graph[edge[0]] = []
        # append edges
        if edge[1]:
            graph[edge[0]].append(edge[1])

    return graph

def get_edge_mask(graph, sub_graph):
    graph_edge_list     = get_edge_list_matrix(graph)
    sub_graph_edge_list = get_edge_list_matrix(sub_graph)
    return np.transpose([[ 1 if edge in sub_graph_edge_list else 0 for edge in graph_edge_list]])

def get_node_mask(graph, sub_graph):
    graph_node_list     = get_node_list_matrix(graph)
    sub_graph_node_list = get_node_list_matrix(sub_graph)
    return np.array([ 1 if node in sub_graph_node_list else 0 for node in graph_node_list])

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MATRIX TEMPLATE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def _matrix(graph, weight_in, weight_out, node_list=[], edge_list=[]):

    # list of nodes and edges
    if not node_list:
        node_list = get_node_list_matrix(graph)
    if not edge_list:
        edge_list = get_edge_list_matrix(graph)

    # create matrix
    matrix = np.zeros( shape=( len(edge_list), len(node_list) ), dtype=float )

    # nodes in and out
    nodes_in  = graphs.get_input_nodes(graph, allow_multiport=True)
    nodes_out = graphs.get_output_nodes(graph, allow_multiport=True)

    # input connections
    for node_in in nodes_in:
        edges_in  = get_edges_in(node_in,edge_list)
        for edge_in in edges_in:
            if edge_in[0] in graph[node_in]:
                matrix[edge_list.index(edge_in), node_list.index('input')] =  weight_in(graph, node_in, list(graph.predecessors(edge_in[0])).index(node_in))
                matrix[edge_list.index(edge_in), node_list.index(node_in)] = -weight_in(graph, node_in, list(graph.predecessors(edge_in[0])).index(node_in))
            else:
                matrix[edge_list.index(edge_in), node_list.index('input')] =  weight_in(graph, node_in, 0)
                matrix[edge_list.index(edge_in), node_list.index(node_in)] = -weight_in(graph, node_in, 0)

    # output connections
    for node_out in nodes_out:
        edges_out = get_edges_out(node_out,edge_list)
        for edge_out in edges_out:
            if edge_out[1] in graph[node_out]:
                matrix[edge_list.index(edge_out), node_list.index(node_out)] =  weight_out(graph, node_out, list(graph.successors(node_out)).index(edge_out[1]))
                matrix[edge_list.index(edge_out), node_list.index('output')] = -weight_out(graph, node_out, list(graph.successors(node_out)).index(edge_out[1]))
            else:
                matrix[edge_list.index(edge_out), node_list.index(node_out)] =  weight_out(graph, node_out, 0)
                matrix[edge_list.index(edge_out), node_list.index('output')] = -weight_out(graph, node_out, 0)

    # internal connections
    for node in graph.nodes():
        for edge in graph.adj[node]:
            # matrix[edge_list.index((node, edge)), node_list.index(node)] =  weight_out(graph, node, list(graph.successors(node)).index(edge))
            matrix[edge_list.index((node, edge)), node_list.index(node)] =  weight_out(graph, node, 0)
            # matrix[edge_list.index((node, edge)), node_list.index(edge)] = -weight_in(graph, edge, list(graph.predecessors(edge)).index(node))
            matrix[edge_list.index((node, edge)), node_list.index(edge)] = -weight_in(graph, edge, 0)

    return matrix

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CONNECTIONS MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def get_connections_matrix(graph,node_list=[],edge_list=[]): # TODO: Change to get_connection_matrix

    weight_in  = lambda graph, node, edge_index : 1
    weight_out = lambda graph, node, edge_index : 1

    return _matrix(graph,weight_in,weight_out,node_list,edge_list)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
STREAMS MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def get_streams_matrix(graph,node_list=[],edge_list=[]):

    weight_in  = lambda graph, node, edge_index : graph.nodes[node]['hw'].streams_in()
    weight_out = lambda graph, node, edge_index : graph.nodes[node]['hw'].streams_out()

    return _matrix(graph,weight_in,weight_out,node_list,edge_list)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BALANCED STREAMS MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def get_balanced_streams_matrix(graph,node_list=[],edge_list=[]):
    # get streams matrix
    streams_matrix = get_streams_matrix(graph,node_list=node_list,edge_list=edge_list)
    # balance streams
    return np.multiply(streams_matrix,scipy.linalg.null_space(streams_matrix).T)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RATES MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def get_rates_matrix(graph,node_list=[],edge_list=[]):

    weight_in  = lambda graph, node, edge_index : graph.nodes[node]['hw'].rate_in()
    weight_out = lambda graph, node, edge_index : graph.nodes[node]['hw'].rate_out()

    return _matrix(graph,weight_in,weight_out,node_list,edge_list)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BALANCED RATES MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def get_balanced_rates_matrix(graph,node_list=[],edge_list=[]):

    # get connections matrix
    connections_matrix = get_connections_matrix(graph,node_list=node_list,edge_list=edge_list)
    # get the interval of the graph
    interval = np.max(np.absolute(get_interval_matrix(graph,node_list=node_list,edge_list=edge_list)))
    # get streams matrix
    streams_matrix  = get_streams_matrix(graph,node_list=node_list,edge_list=edge_list)
    workload_matrix = get_workload_matrix(graph,node_list=node_list,edge_list=edge_list)
    # return the balanced rates matrix
    return np.nan_to_num(np.divide(
        workload_matrix,
        np.multiply(
            np.multiply(
                interval,
                connections_matrix),
            streams_matrix)))

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WORKLOAD MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def get_workload_matrix(graph,node_list=[],edge_list=[]):

    weight_in  = lambda graph, node, edge_index : graph.nodes[node]['hw'].workload_in()
    weight_out = lambda graph, node, edge_index : graph.nodes[node]['hw'].workload_out()

    return _matrix(graph,weight_in,weight_out,node_list,edge_list)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TOPOLOGY MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def get_topology_matrix(graph,node_list=[],edge_list=[]):
    streams_matrix = get_streams_matrix(graph,node_list=node_list,edge_list=edge_list)
    rates_matrix   = get_balanced_rates_matrix(graph,node_list=node_list,edge_list=edge_list)
    return np.multiply(streams_matrix, np.absolute(rates_matrix))

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
INTERVAL MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def get_interval_matrix(graph,node_list=[],edge_list=[]):
    workload_matrix = get_workload_matrix(graph,node_list=node_list,edge_list=edge_list)
    streams_matrix  = get_streams_matrix(graph,node_list=node_list,edge_list=edge_list)
    rates_matrix    = get_rates_matrix(graph,node_list=node_list,edge_list=edge_list)

    interval_matrix = np.multiply(streams_matrix, np.absolute(rates_matrix))
    interval_matrix = np.divide(workload_matrix, np.absolute(interval_matrix))
    interval_matrix = np.nan_to_num(interval_matrix)

    return np.absolute(interval_matrix)
