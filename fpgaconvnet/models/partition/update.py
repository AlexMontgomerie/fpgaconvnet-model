import copy
import itertools
import json
import math

import fpgaconvnet.tools.graphs as graphs
import networkx as nx
import numpy as np
from fpgaconvnet.models.layers import SqueezeLayer, SqueezeLayer3D
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

MULTIPORT_LAYERS_IN = [ LAYER_TYPE.EltWise, LAYER_TYPE.Concat ]
MULTIPORT_LAYERS_OUT = [ LAYER_TYPE.Split, LAYER_TYPE.Chop ]

def update(self, update_streams=True):

    ## update sizes
    self.wr_layer = self.wr_layer if self.wr_layer != "None" else None
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

    ## update streams in and out
    self.input_nodes = graphs.get_input_nodes(self.graph, allow_multiport=True)
    self.ports_in = len(self.input_nodes)

    self.output_nodes = graphs.get_output_nodes(self.graph, allow_multiport=True)
    self.ports_out = len(self.output_nodes)

    if update_streams:
        self.streams_in = [self.graph.nodes[input_node]['hw'].streams_in() for input_node in self.input_nodes]
        self.streams_out = [self.graph.nodes[output_node]['hw'].streams_out() for output_node in self.output_nodes]

    # ## update buffer depths
    for node in self.graph.nodes:
        if self.graph.nodes[node]["type"] in MULTIPORT_LAYERS_IN:
            self.update_multiport_buffer_depth(node)

def update_multiport_buffer_depth(self, multiport_node):

    # check the eltwise node is actually eltwise
    assert self.graph.nodes[multiport_node]["type"] in MULTIPORT_LAYERS_IN, \
            "node does not have multiple ports in"

    # find the lowest common (single) ancestor of the multiport node

    ## get all pairs of inputs to the node
    multiport_prev_nodes = graphs.get_prev_nodes(self.graph, multiport_node)
    multiport_prev_nodes_pairs = list(itertools.combinations(multiport_prev_nodes, 2))

    ## get all the common ancestors for the node pairs
    common_ancestors = [ x[1] for x in nx.all_pairs_lowest_common_ancestor(
        self.graph, multiport_prev_nodes_pairs) ]

    ## topological sort common ancestors and choose the oldest
    sorted_graph_nodes = list(nx.topological_sort(self.graph))
    split_nodes = sorted(common_ancestors, key=lambda n: sorted_graph_nodes.index(n))

    # there is no split node, probably the eltwise/concat node is the first node
    if not split_nodes:
        for idx in range(self.graph.nodes[multiport_node]["hw"].ports_in):
            self.graph.nodes[multiport_node]["hw"].buffer_depth[idx] = 2
        return

    split_node = split_nodes[0]

    # cannot find split layer, maybe it is vertical split
    if not self.graph.nodes[split_node]["type"] in MULTIPORT_LAYERS_OUT:
        #TODO: Check if need to update the buffer depth in this case as well
        return

    # get all the paths split layer and eltwise layer
    all_paths = list(nx.all_simple_paths(self.graph,
        source=split_node, target=multiport_node))

    # initiation interval of the hardware
    # interval = self.get_interval()

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

        # get the rate in
        # rate_in = [ n.rate_in() for n in node_hw ]

        # get the pipeline depth of each node
        node_depth = [ n.pipeline_depth() for n in node_hw ]

        # get the path depth
        path_depths[i] = sum(node_depth) + sum([ (latency[j]/size_in[j]) * \
                np.prod([ size_in[k]/size_out[k] for k in range(j+1)
                    ]) for j in range(len(node_hw)) ])

    # get the longest depths for each prev node
    path_depths_max = { n: [] for n in multiport_prev_nodes }
    for i, path in enumerate(all_paths):
        path_depths_max[path[-2]].append(path_depths[i])
    path_depths_max = { n: max(path_depths_max[n]) for n in multiport_prev_nodes }

    # get the overall max depth
    max_depth = max(path_depths_max.values())

    # update the buffer depths for eltwise layer
    for n, depth in path_depths_max.items():

        # get the input index
        for m in self.graph.nodes[n]["onnx_output"]:
            if m in self.graph.nodes[multiport_node]["onnx_input"]:
                idx = self.graph.nodes[multiport_node]["onnx_input"].index(m)
                break

        # buffer depth is difference of max depth with the paths depth
        if self.graph.nodes[multiport_node]["type"] == LAYER_TYPE.EltWise:
            self.graph.nodes[multiport_node]["hw"].buffer_depth[idx] = \
                    math.ceil(max_depth - depth) + 64
        if self.graph.nodes[multiport_node]["type"] == LAYER_TYPE.Concat:
            n = self.graph.nodes[multiport_node]["hw"]
            extra_cycles = int(sum([ n.channels_in(i)/n.rate_in(i) \
                    for i in range(n.ports_in) ]))
            self.graph.nodes[multiport_node]["hw"].buffer_depth[idx] = \
                    math.ceil(max_depth - depth) + extra_cycles + 64

def reduce_squeeze_fanout(self):
    """
    method to change the parallelism of modules between convolution layers to
    reduce the fanout of the squeeze module. This improves the frequency of the
    design.
    """

    def _add_dummy_squeeze():
        inputs = graphs.get_input_nodes(self.graph, allow_multiport=True)
        for input_node in inputs:
            # add node to graph
            new_node  = "_".join([input_node,"squeeze"])
            # add node to node info
            if self.dimensionality == 2:
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
            elif self.dimensionality == 3:
                self.graph.add_node(new_node,
                    type=LAYER_TYPE.Squeeze,
                    onnx_node=self.graph.nodes[input_node]["onnx_node"],
                    hw=SqueezeLayer3D(
                        self.graph.nodes[input_node]['hw'].rows_in(),
                        self.graph.nodes[input_node]['hw'].cols_in(),
                        self.graph.nodes[input_node]['hw'].depth_in(),
                        self.graph.nodes[input_node]['hw'].channels_in(),
                        self.max_streams_in//len(inputs),
                        self.max_streams_in//len(inputs)
                    )
                )
            # add edge to graph
            self.graph.add_edge(new_node,input_node)
        # check difference in output streams
        outputs = graphs.get_output_nodes(self.graph, allow_multiport=True)
        for output_node in outputs:
            # add node to graph
            new_node  = "_".join(["squeeze",output_node])
            # add node to node info
            if self.dimensionality == 2:
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
            elif self.dimensionality == 3:
                self.graph.add_node(new_node,
                    type=LAYER_TYPE.Squeeze,
                    onnx_node=self.graph.nodes[output_node]["onnx_node"],
                    hw=SqueezeLayer3D(
                        self.graph.nodes[output_node]['hw'].rows_out(),
                        self.graph.nodes[output_node]['hw'].cols_out(),
                        self.graph.nodes[output_node]['hw'].depth_out(),
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
