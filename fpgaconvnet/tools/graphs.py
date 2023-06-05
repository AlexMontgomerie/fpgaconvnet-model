import numpy as np
import scipy
import copy
from numpy.linalg import matrix_rank
import networkx as nx
from networkx.algorithms.dag import ancestors
from networkx.algorithms.dag import descendants
import os
import pydot
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def print_graph(graph):
    for node, edges in graph.adjacency():
        edges = list(edges)
        print(f"{node}\t: {edges}")

def get_input_nodes(graph):
    return [ edge for edge, deg in graph.in_degree() if not deg ]

def get_output_nodes(graph):
    return [ edge for edge, deg in graph.out_degree() if not deg ]

def get_next_nodes(graph, node):
    return list(graph.successors(node))

def get_prev_nodes(graph, node):
    return list(graph.predecessors(node))

def get_next_nodes_all(graph, node):
    return list(descendants(graph, node))

def get_prev_nodes_all(graph, node):
    return list(ancestors(graph, node))

def ordered_node_list(graph): # TODO: make work for parallel networks
    return list( nx.topological_sort(graph) )

def split_graph_horizontal(graph,edge):
    prev_nodes = get_prev_nodes_all(graph,edge[1])
    next_nodes = get_next_nodes_all(graph,edge[0])
    for node in prev_nodes:
        for nnode in get_next_nodes_all(graph, node):
            if nnode not in next_nodes and nnode not in prev_nodes:
                prev_nodes.append(nnode)
    prev_graph = graph.subgraph(prev_nodes).copy()
    next_graph = graph.subgraph(next_nodes).copy()
    return prev_graph, next_graph

def split_graph_vertical(graph, nodes):
    input_node = get_input_nodes(graph)[0]
    output_node = get_output_nodes(graph)[0]
    # find left side graph
    left_nodes = [input_node, nodes[0][0]]
    for node in nodes[0]:
        left_nodes.extend( get_next_nodes_all(graph,node) )
    # find right side graph
    right_nodes = [nodes[1][0]]
    for node in nodes[1]:
        right_nodes.extend( get_next_nodes_all(graph,node) )
    # put output node in the smaller graph
    left_nodes = [node for node in left_nodes if node != output_node]
    right_nodes = [node for node in right_nodes if node != output_node]
    if len(left_nodes) > len(right_nodes):
        right_nodes.append(output_node)
    else:
        left_nodes.append(output_node)
    left_graph = graph.subgraph(left_nodes).copy()
    right_graph = graph.subgraph(right_nodes).copy()
    return left_graph, right_graph

def merge_graphs_horizontal(graph_prev, graph_next):
    graph = nx.compose(graph_prev,graph_next)
    prev_output_node = get_output_nodes(graph_prev)[0]
    next_input_node  = get_input_nodes(graph_next)[0]
    graph.add_edge(prev_output_node, next_input_node)
    return graph

def merge_graphs_vertical(graph_prev, graph_next):
    pass

def to_json(graph_in):
    graph_out = nx.DiGraph()
    for node in ordered_node_list(graph_in):
        graph_out.add_node(node.replace("/","_"))
        for edge in get_next_nodes(graph_in, node):
            graph_out.add_edge(node.replace("/","_"),edge.replace("/","_"))
    return nx.jit_data(graph_out)

def from_json(data):
    return nx.jit_graph(data,create_using=nx.DiGraph())

def view_graph(graph,filepath):
    _, name = os.path.split(filepath)
    g = pydot.Dot(graph_type='digraph')
    g.set_node_defaults(shape='record')
    for node in graph:
        if graph.nodes[node]['type'] == LAYER_TYPE.Concat:
            layer_info = graph.nodes[node]['hw'].layer_info()
            node_type  = layer_info['type']
            rows       = layer_info['rows']
            cols       = layer_info['cols']
            channels   = str(layer_info['channels'])
        else:
            layer_info = graph.nodes[node]['hw'].layer_info()
            node_type  = layer_info['type']
            rows       = layer_info['rows']
            cols       = layer_info['cols']
            channels   = layer_info['channels']
        g.add_node(pydot.Node(node,
            label="{{ {node}|type: {type} \n dim: [{rows}, {cols}, {channels}]  }}".format(
            node=node,
            type=node_type,
            rows=rows,
            cols=cols,
            channels=channels)))
        for edge in graph[node]:
            #g.add_edge(pydot.Edge(node,edge,splines="ortho"))
            g.add_edge(pydot.Edge(node,edge,splines="line"))
    g.write_png('outputs/images/'+name+'.png')
