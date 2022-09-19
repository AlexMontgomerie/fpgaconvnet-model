from graphviz import Digraph
import pydot
import os
import random
import copy
import importlib

import onnx
import onnx.utils
import onnx.numpy_helper
import networkx as nx

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.onnx_helper as onnx_helper

from fpgaconvnet.models.layers import BatchNormLayer
# from fpgaconvnet.models.layers import ConvolutionLayer
from fpgaconvnet.models.layers import InnerProductLayer
from fpgaconvnet.models.layers import PoolingLayer
from fpgaconvnet.models.layers import ReLULayer

from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type

def remove_node(graph, node): # TODO: move to tools.graphs
    prev_nodes = graphs.get_prev_nodes(graph,node)
    next_nodes = graphs.get_next_nodes(graph,node)
    graph.remove_node(node)
    for prev_node in prev_nodes:
        for next_node in next_nodes:
            graph.add_edge(prev_node,next_node)

def filter_node_types(graph, layer_type):
    remove_nodes = []
    for node in graph.nodes():
        if graph.nodes[node]['type'] == layer_type:
            remove_nodes.append(node)
    for node in remove_nodes:
        remove_node(graph,node)

def build_graph(model):
    # graph structure
    graph = nx.DiGraph()
    # add all nodes from network
    for node in model.graph.node:
        # get name of node
        name = onnx_helper._name(node)
        # add node to graph
        graph.add_node( name, type=from_onnx_op_type(node.op_type), hw=None, inputs={} )

        if from_onnx_op_type(node.op_type) in [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:
            graph.nodes[name]['inputs'] = { "weights": "", "bias": "" }
    # add all edges from network
    edges = []
    for name in graph.nodes():
        # get node from model
        node = onnx_helper.get_model_node(model, name)
        # add edges into node
        for input_node in node.input:
            # add initializers
            if onnx_helper.get_model_initializer(model, input_node) is not None:
                # get input details
                input_details = onnx_helper.get_model_input(model, input_node)
                # convolution inputs
                if graph.nodes[name]["type"] == LAYER_TYPE.Convolution:
                    if len(input_details.type.tensor_type.shape.dim) == 4:
                        graph.nodes[name]['inputs']['weights'] = input_node
                    if len(input_details.type.tensor_type.shape.dim) == 1:
                        graph.nodes[name]['inputs']['bias'] = input_node
                # inner product inputs
                if graph.nodes[name]["type"] == LAYER_TYPE.InnerProduct:
                    if len(input_details.type.tensor_type.shape.dim) == 2:
                        graph.nodes[name]['inputs']['weights'] = input_node
                    if len(input_details.type.tensor_type.shape.dim) == 1:
                        graph.nodes[name]['inputs']['bias'] = input_node
                continue
            input_node = onnx_helper._format_name(input_node)
            if input_node != name:
                edges.append((input_node, name))
        # add eges out of node
        for output_node in node.output:
            output_node = onnx_helper._format_name(output_node)
            if output_node in graph.nodes():
                if output_node != name:
                    edges.append((name,output_node))
    # add edges to graph
    for edge in edges:
        graph.add_edge(*edge)
    # return graph
    return graph

def add_hardware(model, graph, data_width=16, weight_width=8,
        biases_width=16, acc_width=30, backend="hls"):

    # import layers
    convolution = importlib.import_module(f"fpgaconvnet.models.layers.{backend}")

    # iterate over nodes in graph
    for node in model.graph.node:
        # get node name
        name = onnx_helper._name(node)
        # check if node in graph
        if not name in graph.nodes():
            continue
        # Convolution layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Convolution:
            # get number of filters
            weights_input = graph.nodes[name]["inputs"]["weights"]
            weights_dim = onnx_helper.get_model_input(model,weights_input)
            filters = int(weights_dim.type.tensor_type.shape.dim[0].dim_value)
            # get node attributes
            attr = onnx_helper._format_attr(node.attribute)
            # default attributes
            attr.setdefault("group", 1)
            attr.setdefault("strides", [1,1])
            attr.setdefault("pads", [0,0,0,0])
            attr.setdefault("dilations", [1,1])
            # check for bias
            has_bias = 0
            if graph.nodes[name]["inputs"]["bias"] != "": # no bias
                has_bias = 1
            # create convolution layer hardware
            graph.nodes[name]['hw'] = convolution.ConvolutionLayer(
                filters,
                0, # initialise rows to 0
                0, # initialise cols to 0
                0, # initialise channels to 0
                kernel_size =attr["kernel_shape"],
                stride =attr["strides"],
                pad =attr["pads"],
                groups =attr["group"],
                has_bias = has_bias
            )
            continue
        # FC Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.InnerProduct:
            # get number of filters
            weights_input = graph.nodes[name]["inputs"]["weights"]
            weights_dim = onnx_helper.get_model_input(model,weights_input)
            filters = int(weights_dim.type.tensor_type.shape.dim[0].dim_value)
            # check for bias
            has_bias = 0
            if graph.nodes[name]["inputs"]["bias"] != "": # no bias
                has_bias = 1
            # create inner product layer hardware
            graph.nodes[name]['hw'] = InnerProductLayer(
                filters,
                0, # initialise rows to 0
                0, # initialise cols to 0
                0, # initialise channels to 0
                has_bias = has_bias
            )
            continue
        # Pooling layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Pooling:
            # get node attributes
            attr = onnx_helper._format_attr(node.attribute)
            # default attributes
            attr.setdefault("strides", [1,1])
            attr.setdefault("pads", [0,0,0,0])
            attr.setdefault("dilations", [1,1])
            # create pooling layer hardware
            graph.nodes[name]['hw'] = PoolingLayer(
                0, # initialise rows to 0
                0, # initialise cols to 0
                0, # initialise channels to 0
                pool_type = 'max', # TODO: change so that it does AVG also
                kernel_size =attr["kernel_shape"],
                stride =attr["strides"],
                pad =attr["pads"],
            )
            continue
        # ReLU Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.ReLU:
            # create relu layer hardware
            graph.nodes[name]['hw'] = ReLULayer(
                0, # initialise rows to 0
                0, # initialise cols to 0
                0, # initialise channels to 0
            )
            continue
        # BatchNorm Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.BatchNorm:
            graph.nodes[name]['hw'] = BatchNormLayer(
                0, # initialise rows to 0
                0, # initialise cols to 0
                0, # initialise channels to 0
                1, # initialise coarse in to 0
                1, # initialise coarse out to 0
            )
            continue
        raise NameError(f"{name}: type {str(graph.nodes[name]['type'])} does not exist!")

def add_dimensions(model, graph):
    # add input dimensions
    if len(model.graph.input[0].type.tensor_type.shape.dim) <= 2:
        input_channels  = int(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
        input_rows      = 1
        input_cols      = 1
    else:
        input_channels  = int(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
        input_rows      = int(model.graph.input[0].type.tensor_type.shape.dim[2].dim_value)
        input_cols      = int(model.graph.input[0].type.tensor_type.shape.dim[3].dim_value)
    # update input node hardware
    input_node = graphs.get_input_nodes(graph)[0]
    graph.nodes[input_node]['hw'].channels  = input_channels
    graph.nodes[input_node]['hw'].rows      = input_rows
    graph.nodes[input_node]['hw'].cols      = input_cols
    # iterate over layers in model
    nodes = list(graph.nodes())
    nodes.remove(input_node)
    for node in nodes:
        # find previous node
        prev_nodes = graphs.get_prev_nodes(graph, node)
        for prev_node in prev_nodes: # TODO: support parallel networks
            # get previous node output dimensions
            dim = onnx_helper._out_dim(model, prev_node)
            # update input dimensions
            graph.nodes[node]['hw'].channels = dim[0]
            graph.nodes[node]['hw'].rows     = dim[1]
            graph.nodes[node]['hw'].cols     = dim[2]

def parse_net(filepath, view=True, data_width=16, weight_width=8,
        biases_width=16, acc_width=30, fuse_bn=True, backend="chisel"):

    # load onnx model
    model = onnx_helper.load(filepath,fuse_bn)

    # get graph
    graph = build_graph(model)

    # remove input node
    remove_nodes = []
    for node in graph.nodes:
        if "type" not in graph.nodes[node]:
            remove_nodes.append(node)
    for node in remove_nodes:
        graph.remove_node(node)

    # remove unnecessary nodes
    remove_layer_types = [
            LAYER_TYPE.Dropout,
            LAYER_TYPE.Transpose,
            LAYER_TYPE.Flatten,
            LAYER_TYPE.Clip,
            LAYER_TYPE.Cast,
            LAYER_TYPE.Squeeze,
            LAYER_TYPE.Shape,
            LAYER_TYPE.Softmax,
            LAYER_TYPE.LRN
    ]
    for layer_type in remove_layer_types:
        filter_node_types(graph, layer_type)

    # add hardware to graph
    add_hardware(model, graph, data_width, weight_width,
            biases_width, acc_width, backend)

    # add layer dimensions
    add_dimensions(model, graph)

    # update all layers
    for node in graph.nodes:
        graph.nodes[node]['hw'].update()

    return model, graph

