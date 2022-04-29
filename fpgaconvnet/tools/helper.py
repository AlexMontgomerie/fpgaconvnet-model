"""
A set of helper functions for the various transforms.
"""

from fpgaconvnet.tools.layer_enum import LAYER_TYPE
from functools import reduce

def get_all_layers(graph, layer_type):
    """
    Parameters
    ----------
    graph: networkx.DiGraph
        graph of a partition

    layer_type: fpgaconvnet.tools.layer_enum.LAYER_TYPE
        type of layer that you want to filter from
        the graph

    Returns
    -------
    list
        A list of layers within the graph with the
        specified layer type
    """
    layers= []
    for node in graph.nodes():
        if graph.nodes[node]['type'] == layer_type:
            layers.append(node)
    return layers

def get_factors(n):
    """
    Parameters
    ----------
    n: int

    Returns
    -------
    list
        list of integers that are factors of `n`
    """
    return list(set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

