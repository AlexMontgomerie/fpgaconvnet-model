import json
import math
import copy
import itertools

import numpy as np
import networkx as nx

import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def update(self):

    ## remove auxiliary layers
    self.remove_squeeze()

    ## update streams in and out
    input_node  = graphs.get_input_nodes(self.graph)[0]
    output_node = graphs.get_output_nodes(self.graph)[0]

    ## get valid streams in and out
    streams_in_valid = self.graph.nodes[input_node]["hw"].get_coarse_in_feasible()
    streams_out_valid = self.graph.nodes[output_node]["hw"].get_coarse_out_feasible()

    # get the max stream values in and out
    streams_in_max = min(self.max_streams_in, self.graph.nodes[input_node]["hw"].streams_in())
    streams_out_max = min(self.max_streams_out, self.graph.nodes[output_node]["hw"].streams_out())

    # choose the max of all the valid stream values, below the max
    self.streams_in = max([ s for s in streams_in_valid if s <= streams_in_max ])
    self.streams_out = max([ s for s in streams_out_valid if s <= streams_out_max ])

    ## add auxiliary layers
    self.add_squeeze()

    ## update streams in and out
    self.input_nodes = graphs.get_input_nodes(self.graph)
    self.output_nodes = graphs.get_output_nodes(self.graph)

    ## update sizes
    self.wr_layer = self.wr_layer if self.wr_layer != "None" else None
    self.size_in  = self.graph.nodes[self.input_nodes[0]]['hw'].size_in()
    self.size_out = self.graph.nodes[self.input_nodes[0]]['hw'].size_out()
    if self.wr_layer != None:
        self.size_wr = self.graph.nodes[self.wr_layer]['hw'].get_parameters_size()['weights']
    else:
        self.size_wr = 0

    ## update the modules
    for node in self.graph.nodes:
        self.graph.nodes[node]["hw"].update()

    ## add auxiliary layers
    self.remove_squeeze()
    self.add_squeeze()

    ## update buffer depths
    for node in self.graph.nodes:
        if self.graph.nodes[node]["type"] == LAYER_TYPE.EltWise:
            self.update_eltwise_buffer_depth(node)

def update_eltwise_buffer_depth(self, eltwise_node):

    # check the eltwise node is actually eltwise
    assert self.graph.nodes[eltwise_node]["type"] == LAYER_TYPE.EltWise, "node is not of type EltWise"

    # search back in the graph for the split layer
    split_node = eltwise_node
    while self.graph.in_degree(split_node) > 0:
        split_node = graphs.get_prev_nodes(self.graph, split_node)[0]
        if self.graph.nodes[split_node]["type"] == LAYER_TYPE.Split:
            break

    # cannot find split layer, maybe it is vertical split
    if self.graph.nodes[split_node]["type"] != LAYER_TYPE.Split:
        return

    # get all the paths split layer and eltwise layer
    all_paths = list(nx.all_simple_paths(self.graph, source=split_node, target=eltwise_node))

    # calculate the depth for each path
    path_depths = [0]*len(all_paths)
    for i, path in enumerate(all_paths):

        # get the hardware model for each node in the path
        node_hw = [ self.graph.nodes[node]["hw"] for node in path ]
        # print([ self.graph.nodes[node]["type"] for node in path ])
        # print([ self.graph.nodes[node]["hw"].size_in() for node in path ])
        # print([ self.graph.nodes[node]["hw"].size_out() for node in path ])
        # print([ self.graph.nodes[node]["hw"].coarse_in for node in path ])
        # print([ self.graph.nodes[node]["hw"].coarse_out for node in path ])

        # get expansion of each node in path
        expansion = [ n.size_in() / n.size_out() for n in node_hw ]

        # get the pipeline depth of each node
        node_depth = [ n.pipeline_depth() for n in node_hw ]

        # get the path depth
        path_depths[i] = node_depth[0] + sum([
            node_depth[j]*np.prod([ expansion[k] for k in range(j) ]) for j in range(1,len(node_hw)) ])

    # get all prev nodes of the eltwise layer
    eltwise_prev_nodes = graphs.get_prev_nodes(self.graph, eltwise_node)

    # update the buffer depths for eltwise layer
    for i, path in enumerate(all_paths):

        # get the input index
        idx = eltwise_prev_nodes.index(path[-2])

        # buffer depth is difference of max depth with the paths depth
        self.graph.nodes[eltwise_node]["hw"].buffer_depth[idx] = math.ceil(max(path_depths) - path_depths[i])

def reduce_squeeze_fanout(self):
    """
    method to change the parallelism of modules between convolution layers to
    reduce the fanout of the squeeze module. This improves the frequency of the
    design.
    """

    # remove all the squeeze nodes
    self.remove_squeeze()

    # get all the convolution and inner product layers in graph
    find_conv = lambda node: self.graph.nodes[node]["type"] in \
            [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]
    all_conv = filter(find_conv, self.graph.nodes)

    # get all pairs of conv modules
    all_conv_pairs = itertools.combinations(all_conv, 2)

    # get all the paths for these pairs
    all_conv_paths = [ nx.all_simple_paths(self.graph,
        source=pair[0], target=pair[1]) for pair in all_conv_pairs ]
    all_conv_paths = [ path for paths in all_conv_paths for path in paths ]

    # filter out paths with only two conv/ inner-prod layers in
    filter_conv_pairs = lambda path: len([ node for node in path if find_conv(node) ]) == 2
    conv_paths = filter(filter_conv_pairs, all_conv_paths)

    # iterate over the conv paths
    for path in conv_paths:

        # get nodes in and out
        node_in = path[0]
        node_out = path[-1]

        # get coarse between conv layers
        coarse_start = self.graph.nodes[node_in]["hw"].coarse_out
        coarse_end = self.graph.nodes[node_out]["hw"].coarse_in

        # choose the min of coarse factors
        coarse_between = min(coarse_start, coarse_end)

        # iterate over the nodes inbetween
        for node in path[1:-1]:

            # update the coarse factor for in-between nodes
            self.graph.nodes[node_out]["hw"].coarse = coarse_between
            self.graph.nodes[node]["hw"].update()

    # add back the squeeze modules
    self.add_squeeze()
