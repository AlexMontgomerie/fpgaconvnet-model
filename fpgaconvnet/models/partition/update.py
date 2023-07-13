import json
import math
import copy
import itertools

import numpy as np
import networkx as nx

import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE
from fpgaconvnet.models.layers import SqueezeLayer

MULTIPORT_LAYERS = [ LAYER_TYPE.EltWise, LAYER_TYPE.Concat ]

def update(self):

    ## remove auxiliary layers
    self.remove_squeeze()

    ## update streams in
    self.streams_in = []
    inputs = graphs.get_input_nodes(self.graph)
    for i, input_node in enumerate(inputs):
        ## get valid streams in
        streams_in_valid = self.graph.nodes[input_node]["hw"].get_coarse_in_feasible()
        # get the max stream values in
        streams_in_max = min(self.max_streams_in//len(inputs), self.graph.nodes[input_node]["hw"].streams_in())
        # choose the max of all the valid stream values, below the max
        self.streams_in.append(max([ s for s in streams_in_valid if s <= streams_in_max ]))

    ## update streams out
    self.streams_out = []
    outputs = graphs.get_output_nodes(self.graph)
    for i, output_node in enumerate(outputs):
        ## get valid streams out
        streams_out_valid = self.graph.nodes[output_node]["hw"].get_coarse_out_feasible()
        # get the max stream values out
        streams_out_max = min(self.max_streams_out//len(outputs), self.graph.nodes[output_node]["hw"].streams_out())
        # choose the max of all the valid stream values, below the max
        self.streams_out.append(max([ s for s in streams_out_valid if s <= streams_out_max ]))

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
        if self.graph.nodes[node]["type"] in MULTIPORT_LAYERS:
            self.update_multiport_buffer_depth(node)

def update_multiport_buffer_depth(self, multiport_node):

    # check the eltwise node is actually eltwise
    assert self.graph.nodes[multiport_node]["type"] in MULTIPORT_LAYERS, \
            "node does not have multiple ports in"

    # search back in the graph for the split layer
    split_node = multiport_node
    while self.graph.in_degree(split_node) > 0:
        split_node = graphs.get_prev_nodes(self.graph, split_node)[0]
        if self.graph.nodes[split_node]["type"] in [ LAYER_TYPE.Split, LAYER_TYPE.Chop ]:
            break

    # cannot find split layer, maybe it is vertical split
    if self.graph.nodes[split_node]["type"] != [ LAYER_TYPE.Split, LAYER_TYPE.Chop ]:
        return

    # get all the paths split layer and eltwise layer
    all_paths = list(nx.all_simple_paths(self.graph, source=split_node, target=multiport_node))

    # calculate the depth for each path
    path_depths = [0]*len(all_paths)
    for i, path in enumerate(all_paths):

        # get the hardware model for each node in the path
        node_hw = [ self.graph.nodes[node]["hw"] for node in path ]

        # get the size in
        size_in = [ n.size_in() for n in node_hw ]

        # get the size out
        size_out = [ n.size_out() for n in node_hw ]

        # get the latency
        latency = [ n.latency() for n in node_hw ]

        # get the pipeline depth of each node
        node_depth = [ n.pipeline_depth() for n in node_hw ]

        # get the path depth
        path_depths[i] = sum(node_depth) + sum([ (latency[j]/size_in[j]) * \
                np.prod([ size_in[k]/size_out[k] for k in range(j+1)
                    ]) for j in range(len(node_hw)) ])

    # get all prev nodes of the eltwise layer
    eltwise_prev_nodes = graphs.get_prev_nodes(self.graph, multiport_node)

    # update the buffer depths for eltwise layer
    for i, path in enumerate(all_paths):

        # get the input index
        idx = eltwise_prev_nodes.index(path[-2])

        # buffer depth is difference of max depth with the paths depth
        # self.graph.nodes[multiport_node]["hw"].buffer_depth[idx] = math.ceil(max(path_depths) - path_depths[i]) + 64
        if self.graph.nodes[multiport_node]["type"] == LAYER_TYPE.EltWise:
            self.graph.nodes[multiport_node]["hw"].buffer_depth[idx] = math.ceil(max(path_depths) - path_depths[i]) + 64
        if self.graph.nodes[multiport_node]["type"] == LAYER_TYPE.Concat:
            n = self.graph.nodes[multiport_node]["hw"]
            extra_cycles = sum([ n.channels_in(i)/n.rate_in(i) for i in range(n.ports_in) ])
            self.graph.nodes[multiport_node]["hw"].buffer_depth[idx] = math.ceil(max(path_depths) - path_depths[i]) + extra_cycles + 64

def reduce_squeeze_fanout(self):
    """
    method to change the parallelism of modules between convolution layers to
    reduce the fanout of the squeeze module. This improves the frequency of the
    design.
    """

    def _add_dummy_squeeze():
        inputs = graphs.get_input_nodes(self.graph)
        for i, input_node in enumerate(inputs):
            # add node to graph
            new_node  = "_".join([input_node,"squeeze"])
            # add node to node info
            self.graph.add_node(new_node,
                type=LAYER_TYPE.Squeeze,
                onnx_node=self.graph.nodes[input_node]["onnx_node"],
                hw=SqueezeLayer(
                    self.graph.nodes[input_node]['hw'].rows_in(),
                    self.graph.nodes[input_node]['hw'].cols_in(),
                    self.graph.nodes[input_node]['hw'].channels_in(),
                    self.max_streams_in//len(inputs),
                    self.max_streams_in//len(inputs)
                )
            )
            # add edge to graph
            self.graph.add_edge(new_node,input_node)
        # check difference in output streams
        outputs = graphs.get_output_nodes(self.graph)
        for i, output_node in enumerate(outputs):
            # add node to graph
            new_node  = "_".join(["squeeze",output_node])
            # add node to node info
            self.graph.add_node(new_node,
                type=LAYER_TYPE.Squeeze,
                onnx_node=self.graph.nodes[output_node]["onnx_node"],
                hw=SqueezeLayer(
                    self.graph.nodes[output_node]['hw'].rows_out(),
                    self.graph.nodes[output_node]['hw'].cols_out(),
                    self.graph.nodes[output_node]['hw'].channels_out(),
                    self.max_streams_out//len(outputs),
                    self.max_streams_out//len(outputs)
                )
            )
            self.graph.add_edge(output_node,new_node)

    self.remove_squeeze()
    _add_dummy_squeeze()

    # get all the convolution and inner product layers in graph
    find_conv = lambda node: self.graph.nodes[node]["type"] in \
            [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct, LAYER_TYPE.Squeeze ]
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
        coarse_start = self.graph.nodes[node_in]["hw"].streams_out()
        coarse_end = self.graph.nodes[node_out]["hw"].streams_in()

        # choose the min of coarse factors
        coarse_between = min(coarse_start, coarse_end)

        # iterate over the nodes inbetween
        for node in path[1:-1]:

            # update the coarse factor for in-between nodes
            if self.graph.nodes[node]["hw"]._coarse < coarse_between:
                self.graph.nodes[node]["hw"].coarse_in = coarse_between
            if self.graph.nodes[node]["hw"]._coarse < coarse_between:
                self.graph.nodes[node]["hw"].coarse_out = coarse_between
            self.graph.nodes[node]["hw"].update()

    # remove the dummy squeeze nodes
    self.remove_squeeze()
    self.add_squeeze()
