import numpy as np
import copy

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.matrix as matrix

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.layers import SqueezeLayer
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def add_squeeze(self):
    # find mismatching streams
    streams_matrix = matrix.get_streams_matrix(self.graph)
    edge_list = matrix.get_edge_list_matrix(self.graph)
    err = np.sum(streams_matrix,axis=1)
    # iterate over stream difference
    for edge in range(err.shape[0]):
        # mismatch
        if err[edge] != 0:
            # add node to graph
            start_node = edge_list[edge][0]
            end_node   = edge_list[edge][1]
            # new_node   = "_".join([start_name,"squeeze",end_name])
            new_node   = "_".join([start_node,"squeeze",end_node])
            # add node to node info
            self.graph.add_node(new_node,
                type=LAYER_TYPE.Squeeze,
                onnx_node=self.graph.nodes[start_node]["onnx_node"],
                hw=SqueezeLayer(
                    self.graph.nodes[start_node]['hw'].rows_out(),
                    self.graph.nodes[start_node]['hw'].cols_out(),
                    self.graph.nodes[start_node]['hw'].channels_out(),
                    self.graph.nodes[start_node]['hw'].streams_out(),
                    self.graph.nodes[end_node]['hw'].streams_in(),
                    FixedPoint(self.data_width, self.data_width//2)
                )
            )
            # add node to graph
            self.graph.add_edge(start_node,new_node)
            self.graph.add_edge(new_node,end_node)
            self.graph.remove_edge(start_node,end_node)

    # check difference in input streams
    inputs = graphs.get_input_nodes(self.graph)
    for i, input_node in enumerate(inputs):
        if self.streams_in[i] != self.graph.nodes[input_node]['hw'].streams_in():
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
                    self.streams_in[i],
                    self.graph.nodes[input_node]['hw'].streams_in(),
                    FixedPoint(self.data_width, self.data_width//2)
                )
            )
            # add edge to graph
            self.graph.add_edge(new_node,input_node)
    # check difference in output streams
    outputs = graphs.get_output_nodes(self.graph)
    for i, output_node in enumerate(outputs):
        if self.streams_out[i] != self.graph.nodes[output_node]['hw'].streams_out():
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
                    self.graph.nodes[output_node]['hw'].streams_out(),
                    self.streams_out[i]
                )
            )
            self.graph.add_edge(output_node,new_node)

def remove_node_by_type(self, layer_type):
    # get input and output graphs
    input_node  = graphs.get_input_nodes(self.graph)[0]
    output_node = graphs.get_output_nodes(self.graph)[0]
    # remove input squeeze module
    if input_node in self.graph.nodes:
        if self.graph.nodes[input_node]['type'] == layer_type:
            self.graph.remove_node(input_node)
    # remove output squeeze module
    if output_node in self.graph.nodes:
        if self.graph.nodes[output_node]['type'] == layer_type:
            self.graph.remove_node(output_node)
    # remove intermediate squeeze modules
    remove_nodes = []
    for node in self.graph.nodes():
        if self.graph.nodes[node]['type'] == layer_type:
            # add squeeze nodes to list
            remove_nodes.append(node)
            # place edge back
            prev_node = graphs.get_prev_nodes(self.graph,node)[0]
            next_node = graphs.get_next_nodes(self.graph,node)[0]
            self.graph.add_edge(prev_node,next_node)
    # remove squeeze nodes
    self.graph.remove_nodes_from(remove_nodes)

def remove_squeeze(self):
    remove_node_by_type(self, LAYER_TYPE.Squeeze)
    # # get input and output graphs
    # input_node  = graphs.get_input_nodes(self.graph)[0]
    # output_node = graphs.get_output_nodes(self.graph)[0]
    # # remove input squeeze module
    # if input_node in self.graph.nodes:
    #     if self.graph.nodes[input_node]['type'] == LAYER_TYPE.Squeeze:
    #         self.graph.remove_node(input_node)
    # # remove output squeeze module
    # if output_node in self.graph.nodes:
    #     if self.graph.nodes[output_node]['type'] == LAYER_TYPE.Squeeze:
    #         self.graph.remove_node(output_node)
    # # remove intermediate squeeze modules
    # remove_nodes = []
    # for node in self.graph.nodes():
    #     if self.graph.nodes[node]['type'] == LAYER_TYPE.Squeeze:
    #         # add squeeze nodes to list
    #         remove_nodes.append(node)
    #         # place edge back
    #         prev_node = graphs.get_prev_nodes(self.graph,node)[0]
    #         next_node = graphs.get_next_nodes(self.graph,node)[0]
    #         self.graph.add_edge(prev_node,next_node)
    # # remove squeeze nodes
    # self.graph.remove_nodes_from(remove_nodes)
