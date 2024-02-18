import numpy as np
import copy

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.matrix as matrix

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.layer_enum import LAYER_TYPE
from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

def add_squeeze(self):
    # find mismatching streams
    # print("Add Squeeze called")
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
            i = graphs.get_prev_nodes(self.graph,end_node).index(start_node)
            j = graphs.get_next_nodes(self.graph,start_node).index(end_node)
            assert self.graph.nodes[end_node]['hw'].stream_inputs[i] == self.graph.nodes[start_node]['hw'].stream_outputs[j]
            if self.graph.nodes[end_node]['hw'].stream_inputs[i] and self.graph.nodes[start_node]['hw'].stream_outputs[j]:
                # edge is already streaming off-chip
                continue
            new_node   = "_".join([start_node,"squeeze",end_node])

            # create squeeze node hardware config
            config = {
                "rows": self.graph.nodes[start_node]['hw'].rows_out(),
                "cols": self.graph.nodes[start_node]['hw'].cols_out(),
                "channels": self.graph.nodes[start_node]['hw'].channels_out(),
                "coarse_in": self.graph.nodes[start_node]['hw'].streams_out(),
                "coarse_out": self.graph.nodes[end_node]['hw'].streams_in(),
                "data_t": self.graph.nodes[start_node]['hw'].output_t,
                "input_compression_ratio": self.graph.nodes[start_node]['hw'].output_compression_ratio,
                "output_compression_ratio": self.graph.nodes[end_node]['hw'].input_compression_ratio,
            }

            if self.arch.dimensionality == DIMENSIONALITY.THREE:
                config["depth"] = self.graph.nodes[start_node]['hw'].depth_out()

            # add node to node info
            self.graph.add_node(new_node,
                type=LAYER_TYPE.Squeeze,
                onnx_node=self.graph.nodes[start_node]["onnx_node"],
                onnx_input=self.graph.nodes[start_node]["onnx_input"],
                onnx_output=self.graph.nodes[start_node]["onnx_output"],
                hw=LayerBase.build("squeeze", config, self.arch)
            )

            # add node to graph
            self.graph.add_edge(start_node,new_node)
            self.graph.add_edge(new_node,end_node)
            self.graph.remove_edge(start_node,end_node)

    # check difference in input streams
    inputs = graphs.get_input_nodes(self.graph, allow_multiport=True)
    for i, input_node in enumerate(inputs):
        if self.streams_in[i] != self.graph.nodes[input_node]['hw'].streams_in():
            # add node to graph
            new_node  = "_".join([input_node,"squeeze"])

            # create squeeze node hardware config
            config = {
                "rows": self.graph.nodes[input_node]['hw'].rows_in(),
                "cols": self.graph.nodes[input_node]['hw'].cols_in(),
                "channels": self.graph.nodes[input_node]['hw'].channels_in(),
                "coarse_in": self.streams_in[i],
                "coarse_out": self.graph.nodes[input_node]['hw'].streams_in(),
                "data_t": self.graph.nodes[input_node]['hw'].input_t,
            }

            if self.arch.dimensionality == DIMENSIONALITY.THREE:
                config["depth"] = self.graph.nodes[input_node]['hw'].depth_in()

            # add node to node info
            self.graph.add_node(new_node,
                type=LAYER_TYPE.Squeeze,
                onnx_node=self.graph.nodes[input_node]["onnx_node"],
                onnx_input=self.graph.nodes[input_node]["onnx_input"],
                onnx_output=self.graph.nodes[input_node]["onnx_output"],
                hw=LayerBase.build("squeeze", config, self.arch)
            )

            # add edge to graph
            self.graph.add_edge(new_node,input_node)

    # check difference in output streams
    outputs = graphs.get_output_nodes(self.graph, allow_multiport=True)
    for i, output_node in enumerate(outputs):
        if self.streams_out[i] != self.graph.nodes[output_node]['hw'].streams_out():
            # add node to graph
            new_node  = "_".join(["squeeze",output_node])

            # create squeeze node hardware config
            config = {
                "rows": self.graph.nodes[output_node]['hw'].rows_out(),
                "cols": self.graph.nodes[output_node]['hw'].cols_out(),
                "channels": self.graph.nodes[output_node]['hw'].channels_out(),
                "coarse_in": self.graph.nodes[output_node]['hw'].streams_out(),
                "coarse_out": self.streams_out[i],
                "data_t": self.graph.nodes[output_node]['hw'].output_t,
            }

            if self.arch.dimensionality == DIMENSIONALITY.THREE:
                config["depth"] = self.graph.nodes[output_node]['hw'].depth_out()

            # add node to node info
            self.graph.add_node(new_node,
                type=LAYER_TYPE.Squeeze,
                onnx_node=self.graph.nodes[output_node]["onnx_node"],
                onnx_input=self.graph.nodes[output_node]["onnx_input"],
                onnx_output=self.graph.nodes[output_node]["onnx_output"],
                hw=LayerBase.build("squeeze", config, self.arch)
            )
            self.graph.add_edge(output_node,new_node)

def remove_node_by_type(self, layer_type):
    # remove input squeeze module
    for input_node in graphs.get_input_nodes(self.graph, allow_multiport=True):
        if input_node in self.graph.nodes:
            if self.graph.nodes[input_node]['type'] == layer_type:
                self.graph.remove_node(input_node)
    # remove output squeeze module
    for output_node in graphs.get_output_nodes(self.graph, allow_multiport=True):
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
            if graphs.get_prev_nodes(self.graph,node) == []:
                continue
            prev_node = graphs.get_prev_nodes(self.graph,node)[0]
            if graphs.get_next_nodes(self.graph,node) == []:
                continue
            next_node = graphs.get_next_nodes(self.graph,node)[0]
            self.graph.add_edge(prev_node,next_node)
    # remove squeeze nodes
    self.graph.remove_nodes_from(remove_nodes)

def remove_squeeze(self):
    remove_node_by_type(self, LAYER_TYPE.Squeeze)
