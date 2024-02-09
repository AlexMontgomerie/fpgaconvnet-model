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

MULTIPORT_LAYERS_IN = [ LAYER_TYPE.EltWise, LAYER_TYPE.Concat ]
MULTIPORT_LAYERS_OUT = [ LAYER_TYPE.Split, LAYER_TYPE.Chop ]

def print_graph(graph):
    for node, edges in graph.adjacency():
        edges = list(edges)
        print(f"{node}\t: {edges}")

def get_input_nodes(graph, allow_multiport=False):
    if allow_multiport:
        return sorted([ edge for edge, deg in graph.in_degree() if not deg or (graph.nodes[edge]['type'] in MULTIPORT_LAYERS_IN and graph.nodes[edge]['hw'].ports_in > deg) ])
    else:
        return sorted([ edge for edge, deg in graph.in_degree() if not deg ])

def get_output_nodes(graph, allow_multiport=False):
    if allow_multiport:
        return sorted([ edge for edge, deg in graph.out_degree() if not deg or (graph.nodes[edge]['type'] in MULTIPORT_LAYERS_OUT and graph.nodes[edge]['hw'].ports_out > deg) ])
    else:
        return sorted([ edge for edge, deg in graph.out_degree() if not deg ])

def get_next_nodes(graph, node):
    return sorted(list(graph.successors(node)))

def get_prev_nodes(graph, node):
    return sorted(list(graph.predecessors(node)))

def get_next_nodes_all(graph, node):
    return sorted(list(descendants(graph, node)))

def get_prev_nodes_all(graph, node):
    return sorted(list(ancestors(graph, node)))

def get_multiport_layers(graph, direction):
    if direction == "in":
        return [ node for node in ordered_node_list(graph) if graph.nodes[node]['type'] in MULTIPORT_LAYERS_IN ]
    elif direction == "out":
        return [ node for node in ordered_node_list(graph) if graph.nodes[node]['type'] in MULTIPORT_LAYERS_OUT ]
    else:
        raise ValueError("direction must be 'in' or 'out'")

def get_branch_edges_all(partitions):
    network_branch_edges = []
    for i in range(len(partitions)):
        multiport_layers_out = get_multiport_layers(partitions[i].graph, "out")
        for node in multiport_layers_out:
            next_nodes = get_next_nodes(partitions[i].graph, node)
            assert len(next_nodes) > 1, "multiport layer out must have more than one connected nodes at its output"
            for next_node in next_nodes:
                if (node, next_node) not in network_branch_edges:
                    network_branch_edges.append((node, next_node))

        multiport_layers_in = get_multiport_layers(partitions[i].graph, "in")
        for node in multiport_layers_in:
            prev_nodes = get_prev_nodes(partitions[i].graph, node)
            assert len(prev_nodes) > 1, "multiport layer in must have more than one connected nodes at its input"
            for prev_node in prev_nodes:
                if (prev_node, node) not in network_branch_edges and (node, prev_node) not in network_branch_edges:
                    network_branch_edges.append((prev_node, node))
    return network_branch_edges

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
    input_node_type = graph.nodes[input_node]['type']
    output_node = get_output_nodes(graph)[0]
    output_node_type = graph.nodes[output_node]['type']
    # find left side graph
    left_nodes = [input_node, nodes[0][0]]
    for node in nodes[0]:
        left_nodes.extend( get_next_nodes_all(graph,node) )
    # find right side graph
    right_nodes = [nodes[1][0]]
    for node in nodes[1]:
        right_nodes.extend( get_next_nodes_all(graph,node) )
    left_nodes = [node for node in left_nodes if node != output_node]
    right_nodes = [node for node in right_nodes if node != output_node]

    if input_node_type in MULTIPORT_LAYERS_OUT and output_node_type in MULTIPORT_LAYERS_IN:
        # separate input (multiport outputs) and output (multiport inputs) nodes
        if input_node in left_nodes:
            right_nodes.append(output_node)
        else:
            left_nodes.append(output_node)
    else:
        # put output node in the smaller graph
        if len(left_nodes) > len(right_nodes):
            right_nodes.append(output_node)
        else:
            left_nodes.append(output_node)

    left_graph = graph.subgraph(left_nodes).copy()
    right_graph = graph.subgraph(right_nodes).copy()
    return left_graph, right_graph

def merge_graphs_horizontal(graph_prev, graph_next, network_branch_edges):
    graph = nx.compose(graph_prev, graph_next)
    prev_output_nodes = get_output_nodes(graph_prev)
    next_input_nodes  = get_input_nodes(graph_next)

    for edge in network_branch_edges:
        if edge[0] in graph.nodes() and edge[1] in graph.nodes() \
            and edge not in list(graph.edges()):
            print(f"create branch edge {edge}")
            graph.add_edge(edge[0], edge[1])
            if edge[0] in prev_output_nodes:
                prev_output_nodes.remove(edge[0])
            if edge[1] in next_input_nodes:
                next_input_nodes.remove(edge[1])

    if len(prev_output_nodes) > 0 and len(next_input_nodes) > 0:
        print(f"create edge {(prev_output_nodes[0], next_input_nodes[0])}")
        graph.add_edge(prev_output_nodes[0], next_input_nodes[0])

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
