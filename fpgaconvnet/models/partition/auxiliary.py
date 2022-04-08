import numpy as np
import copy

import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.tools.matrix as matrix
import fpgaconvnet_optimiser.tools.onnx_helper as onnx_helper

from fpgaconvnet_optimiser.models.layers import SqueezeLayer

from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

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
            start_name = onnx_helper.gen_layer_name(self.graph, edge_list[edge][0])
            end_name   = onnx_helper.gen_layer_name(self.graph, edge_list[edge][1])
            start_node = edge_list[edge][0]
            end_node   = edge_list[edge][1]
            new_node   = "_".join([start_name,"squeeze",end_name])
            # add node to node info
            self.graph.add_node(new_node,type=LAYER_TYPE.Squeeze,
                hw=SqueezeLayer(
                    self.graph.nodes[start_node]['hw'].rows_out(),
                    self.graph.nodes[start_node]['hw'].cols_out(),
                    self.graph.nodes[start_node]['hw'].channels_out(),
                    self.graph.nodes[start_node]['hw'].streams_out(),
                    self.graph.nodes[end_node]['hw'].streams_in()
                )
            )
            # add node to graph
            self.graph.add_edge(start_node,new_node)
            self.graph.add_edge(new_node,end_node)
            self.graph.remove_edge(start_node,end_node)

    # check difference in input streams
    input_node  = graphs.get_input_nodes(self.graph)[0]
    if self.streams_in != self.graph.nodes[input_node]['hw'].streams_in():
        # add node to graph
        new_node  = "_".join([input_node,"squeeze"])
        # add node to node info
        self.graph.add_node(new_node, type=LAYER_TYPE.Squeeze,
            hw=SqueezeLayer(
                self.graph.nodes[input_node]['hw'].rows_in(),
                self.graph.nodes[input_node]['hw'].cols_in(),
                self.graph.nodes[input_node]['hw'].channels_in(),
                self.streams_in,
                self.graph.nodes[input_node]['hw'].streams_in()
            )
        )
        # add edge to graph
        self.graph.add_edge(new_node,input_node)
    # check difference in output streams
    output_node = graphs.get_output_nodes(self.graph)[0]
    if self.streams_out != self.graph.nodes[output_node]['hw'].streams_out():
        # add node to graph
        new_node  = "_".join(["squeeze",output_node])
        # add node to node info
        self.graph.add_node(new_node,type=LAYER_TYPE.Squeeze,
            hw=SqueezeLayer(
                self.graph.nodes[output_node]['hw'].rows_out(),
                self.graph.nodes[output_node]['hw'].cols_out(),
                self.graph.nodes[output_node]['hw'].channels_out(),
                self.graph.nodes[output_node]['hw'].streams_out(),
                self.streams_out
            )
        )
        self.graph.add_edge(output_node,new_node)

def remove_squeeze(self):
    # get input and output graphs
    input_node  = graphs.get_input_nodes(self.graph)[0]
    output_node = graphs.get_output_nodes(self.graph)[0]
    # remove input squeeze module
    if input_node in self.graph.nodes:
        if self.graph.nodes[input_node]['type'] == LAYER_TYPE.Squeeze:
            self.graph.remove_node(input_node)
    # remove output squeeze module
    if output_node in self.graph.nodes:
        if self.graph.nodes[output_node]['type'] == LAYER_TYPE.Squeeze:
            self.graph.remove_node(output_node)
    # remove intermediate squeeze modules
    remove_nodes = []
    for node in self.graph.nodes():
        if self.graph.nodes[node]['type'] == LAYER_TYPE.Squeeze:
            # add squeeze nodes to list
            remove_nodes.append(node)
            # place edge back
            prev_node = graphs.get_prev_nodes(self.graph,node)[0]
            next_node = graphs.get_next_nodes(self.graph,node)[0]
            self.graph.add_edge(prev_node,next_node)
    # remove squeeze nodes
    self.graph.remove_nodes_from(remove_nodes)

