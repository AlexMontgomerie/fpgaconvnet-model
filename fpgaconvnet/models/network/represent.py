import os
import json

import numpy as np
from google.protobuf.text_format import MessageToString
from google.protobuf.json_format import MessageToJson

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.onnx_helper as onnx_helper
import fpgaconvnet.tools.layer_enum
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def get_model_input_node(self, partition_index):
    input_node = self.partitions[partition_index].input_nodes[0]
    while not onnx_helper.get_model_node(self.model, input_node):
        input_node = graphs.get_next_nodes(self.partitions[partition_index].graph,input_node)[0]
    return onnx_helper.get_model_node(self.model, input_node).input[0]

def get_model_output_node(self, partition_index):
    output_node = self.partitions[partition_index].output_nodes[0]
    while not onnx_helper.get_model_node(self.model, output_node):
        output_node = graphs.get_prev_nodes(self.partitions[partition_index].graph,output_node)[0]
    return onnx_helper.get_model_node(self.model, output_node).output[0]

def save_all_partitions(self,filepath,input_output_from_model=True):
    # create protocol buffer
    partitions = fpgaconvnet_pb2.partitions()
    # iterate over partions
    for i in range(len(self.partitions)):

        # create partition
        partition = partitions.partition.add()

        # add partition info
        partition.id = i
        partition.ports = 1
        if input_output_from_model:
            partition.input_node  = self.get_model_input_node(i) #self.partitions[i]['input_nodes'][0]
            partition.output_node = self.get_model_output_node(i) #self.partitions[i]['output_nodes'][0]
        else:
            partition.input_node  = self.partitions[i].input_nodes[0]
            partition.output_node = self.partitions[i].output_nodes[0]

        # add batch size
        partition.batch_size  = self.partitions[i].batch_size

        # add weights reloading information
        partition.weights_reloading_factor = self.partitions[i].wr_factor
        partition.weights_reloading_layer  = onnx_helper.gen_layer_name(
                    self.partitions[i].graph, self.partitions[i].wr_layer)

        # add all layers (in order)
        for node in graphs.ordered_node_list(self.partitions[i].graph):

            # create layer
            layer = partition.layers.add()
            layer.name = onnx_helper.gen_layer_name(
                    self.partitions[i].graph, node)
            layer.type = fpgaconvnet.tools.layer_enum.to_proto_layer_type(
                    self.partitions[i].graph.nodes[node]['type'])

            # add stream(s) in
            stream_in  = layer.streams_in.add()
            prev_nodes = graphs.get_prev_nodes(self.partitions[i].graph, node)
            if not prev_nodes:
                layer.node_in   = layer.name
                stream_in.name  = "in"
            else :
                prev_layer_name = onnx_helper.gen_layer_name(
                    self.partitions[i].graph, prev_nodes[0])
                layer.node_in   = prev_layer_name
                stream_in.name  = "_".join([prev_layer_name, layer.name])
            stream_in.coarse = self.partitions[i].graph.nodes[node]['hw'].coarse_in

            # add stream(s) out
            stream_out = layer.streams_out.add()
            next_nodes = graphs.get_next_nodes(self.partitions[i].graph, node)
            if not next_nodes:
                layer.node_out  = layer.name
                stream_out.name = "out"
            else:
                next_layer_name = onnx_helper.gen_layer_name(
                    self.partitions[i].graph, next_nodes[0]) # REQUIRED EDIT
                layer.node_out  = next_layer_name
                stream_out.name = "_".join([layer.name, next_layer_name])
            stream_out.coarse = self.partitions[i].graph.nodes[node]['hw'].coarse_out

            # add parameters
            self.partitions[i].graph.nodes[node]['hw'].layer_info(layer.parameters, batch_size=self.partitions[i].batch_size)

            # add weights key
            if self.partitions[i].graph.nodes[node]['type'] in [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:
                layer.weights_path = self.partitions[i].graph.nodes[node]['inputs']['weights']
                layer.bias_path    = self.partitions[i].graph.nodes[node]['inputs']['bias']

    # save in JSON format
    with open(filepath, "w") as f:
        f.write(MessageToJson(partitions,preserving_proto_field_name=True))
        #json.dump(MessageToJson(partitions),f)
