import os
import json
import inspect

import numpy as np
from google.protobuf.text_format import MessageToString
from google.protobuf.json_format import MessageToJson

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.parser.onnx.helper as onnx_helper
import fpgaconvnet.tools.layer_enum
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.models.layers import Layer, Layer3D, MultiPortLayer, MultiPortLayer3D

def get_model_input_nodes(self, i):
    input_node = self.partitions[i].input_nodes[0]
    onnx_input_node = self.partitions[i].graph.nodes[input_node]["onnx_node"]
    while not onnx_helper.get_model_node(self.model, onnx_input_node):
        input_node = graphs.get_next_nodes(
                self.partitions[i].graph,input_node)[0]
        onnx_input_node = self.partitions[i].graph.nodes[input_node]["onnx_node"]
    return onnx_helper.get_model_node(self.model, onnx_input_node).input

def get_model_output_nodes(self, i):
    output_node = self.partitions[i].output_nodes[0]
    onnx_output_node = self.partitions[i].graph.nodes[output_node]["onnx_node"]
    while not onnx_helper.get_model_node(self.model, onnx_output_node):
        output_node = graphs.get_prev_nodes(
                self.partitions[i].graph,output_node)[0]
        onnx_output_node = self.partitions[i].graph.nodes[output_node]["onnx_node"]
    return onnx_helper.get_model_node(self.model, onnx_output_node).output

def get_stream_in_coarse(self, node_hw, index):
    node_base_type = inspect.getmro(type(node_hw))[-2]
    if node_base_type in [ Layer, Layer3D ]:
        return node_hw.streams_in()
    elif node_base_type in [ MultiPortLayer, MultiPortLayer3D ]:
        return node_hw.streams_in(index)
    else:
        raise NotImplementedError(f"base type {node_base_type}")

def get_stream_out_coarse(self, node_hw, index):
    node_base_type = inspect.getmro(type(node_hw))[-2]
    if node_base_type in [ Layer, Layer3D ]:
        return node_hw.streams_out()
    elif node_base_type in [ MultiPortLayer, MultiPortLayer3D ]:
        return node_hw.streams_out(index)
    else:
        raise NotImplementedError

def get_buffer_depth_in(self, node_hw, index):
    node_base_type = inspect.getmro(type(node_hw))[-2]
    if node_base_type in [ Layer, Layer3D ]:
        return node_hw.buffer_depth
    elif node_base_type in [ MultiPortLayer, MultiPortLayer3D ]:
        return node_hw.buffer_depth[index]
    else:
        raise NotImplementedError

def save_all_partitions(self, filepath, input_output_from_model=True):

    # create protocol buffer
    partitions = fpgaconvnet_pb2.partitions()

    # iterate over partions
    for i in range(len(self.partitions)):

        # create partition
        partition = partitions.partition.add()

        # add partition info
        partition.id = i
        partition.ports = 1 # TODO
        if input_output_from_model:
            partition.input_nodes.extend(self.get_model_input_nodes(i))
            partition.output_nodes.extend(self.get_model_output_nodes(i))
        else:
            partition.input_node  = self.partitions[i].input_nodes[0]
            partition.output_node = self.partitions[i].output_nodes[0]

        partition.batch_size  = self.partitions[i].batch_size
        partition.weights_reloading_factor = self.partitions[i].wr_factor
        partition.weights_reloading_layer  = str(self.partitions[i].wr_layer)

        # add all layers (in order)
        for node in graphs.ordered_node_list(self.partitions[i].graph):

            # create layer
            layer = partition.layers.add()
            # layer.name = onnx_helper.format_onnx_name(node)
            layer.name = node
            if self.partitions[i].graph.nodes[node]['type'] == [ LAYER_TYPE.ReLU, LAYER_TYPE.Sigmoid, LAYER_TYPE.SiLU ]:
                self.partitions[i].graph.nodes[node]['type'] = LAYER_TYPE.ReLU
            layer.type = fpgaconvnet.tools.layer_enum.to_proto_layer_type(
                    self.partitions[i].graph.nodes[node]['type'])
            layer.onnx_node = self.partitions[i].graph.nodes[node]['onnx_node']

            # nodes into layer
            prev_nodes = graphs.get_prev_nodes(self.partitions[i].graph, node)

            if not prev_nodes:
                stream_in = layer.streams_in.add()
                stream_in.name  = "in"
                stream_in.coarse = self.get_stream_in_coarse(
                        self.partitions[i].graph.nodes[node]['hw'], 0)
                stream_in.node = node
                stream_in.buffer_depth = self.get_buffer_depth_in(
                        self.partitions[i].graph.nodes[node]['hw'], 0)
            else :
                for j, prev_node in enumerate(prev_nodes):
                    stream_in = layer.streams_in.add()
                    stream_in.name  = "_".join([prev_node, layer.name])
                    stream_in.coarse = self.get_stream_in_coarse(
                            self.partitions[i].graph.nodes[node]['hw'], j)
                    stream_in.node = prev_node
                    stream_in.buffer_depth = self.get_buffer_depth_in(
                            self.partitions[i].graph.nodes[node]['hw'], j)

            # nodes out of layer
            next_nodes = graphs.get_next_nodes(self.partitions[i].graph, node)

            if not next_nodes:
                stream_out = layer.streams_out.add()
                stream_out.name  = "out"
                stream_out.coarse = self.get_stream_out_coarse(
                        self.partitions[i].graph.nodes[node]['hw'], 0)
                stream_out.node = node
            else:
                for j, next_node in enumerate(next_nodes):
                    stream_out = layer.streams_out.add()
                    stream_out.name = "_".join([layer.name, next_node])
                    stream_out.coarse = self.get_stream_out_coarse(
                            self.partitions[i].graph.nodes[node]['hw'], j)
                    stream_out.node = next_node

            # add parameters
            self.partitions[i].graph.nodes[node]['hw'].layer_info(
                    layer.parameters, batch_size=self.partitions[i].batch_size)

            # add weights key
            if self.partitions[i].graph.nodes[node]['type'] in \
                    [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:
                layer.weights_path = self.partitions[i].graph.nodes[node]['inputs']['weights']
                layer.bias_path    = self.partitions[i].graph.nodes[node]['inputs']['bias']

    # save in JSON format
    with open(filepath,"w") as f:
        f.write(MessageToJson(partitions, preserving_proto_field_name=True))

