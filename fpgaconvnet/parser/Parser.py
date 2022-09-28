import copy
import importlib
import os
import random

from graphviz import Digraph
import networkx as nx
import onnx
import onnx.numpy_helper
import onnx.utils
import onnxoptimizer as optimizer
import pydot
import numpy as np

from fpgaconvnet.models.layers import BatchNormLayer
from fpgaconvnet.models.layers import InnerProductLayer
from fpgaconvnet.models.layers import PoolingLayer
from fpgaconvnet.models.layers import ReLULayer

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.onnx_helper as onnx_helper

from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type

class ParseOnnxNode:

    def __init__(self, graph, n, backend="hls"):

        # backend string
        self.backend = backend

        # get name of node
        self.name = onnx_helper._name(n)

        # get the layer type
        self.layer_type = from_onnx_op_type(n.op_type)

        # get inputs and outputs
        all_tensors = [ *graph.input, *graph.output, *graph.value_info ]
        self.inputs = [ next(filter(lambda x: x.name == i, all_tensors)) for i in n.input ]
        self.outputs = [ next(filter(lambda x: x.name == i, all_tensors)) for i in n.output]

        # input and output shape
        self.input_shape = [ x.dim_value for x in self.inputs[0].type.tensor_type.shape.dim ]
        self.output_shape = [ x.dim_value for x in self.outputs[0].type.tensor_type.shape.dim ]

        # get attributes
        self.attr = onnx_helper._format_attr(n.attribute)

    def get_hardware(self):
        raise TypeError(f"{self.layer_type} not implemented!")

    def get_node_info(self):
        return {
            "type" : self.layer_type,
            "hw" : self.get_hardware()
        }

class ParseOnnxConvNode(ParseOnnxNode):

    def get_hardware(self):

        # import layers
        convolution = importlib.import_module(
                f"fpgaconvnet.models.layers.{self.backend}")

        # default attributes
        self.attr.setdefault("group", 1)
        self.attr.setdefault("strides", [1,1])
        self.attr.setdefault("pads", [0,0,0,0])
        self.attr.setdefault("dilations", [1,1])

        # return hardware
        return convolution.ConvolutionLayer(
            self.output_shape[1],
            self.input_shape[2],
            self.input_shape[3],
            self.input_shape[1],
            kernel_size = self.attr["kernel_shape"],
            stride = self.attr["strides"],
            pad = self.attr["pads"],
            groups = self.attr["group"],
            has_bias = len(self.inputs) == 3
        )

class ParseOnnxInnerProductNode(ParseOnnxNode):

    def get_hardware(self):

        # default attributes
        self.attr.setdefault("group", 1)
        self.attr.setdefault("strides", [1,1])
        self.attr.setdefault("pads", [0,0,0,0])
        self.attr.setdefault("dilations", [1,1])

        # return hardware
        return InnerProductLayer(
            self.output_shape[1],
            1, 1,
            np.prod(self.input_shape),
            has_bias = len(self.inputs) == 3
        )


class ParseOnnxReLUNode(ParseOnnxNode):

    def get_hardware(self):

        # return hardware
        return ReLULayer(
            self.input_shape[2],
            self.input_shape[3],
            self.input_shape[1],
        )

class ParseOnnxPoolingNode(ParseOnnxNode):

    def get_hardware(self):

        # default attributes
        self.attr.setdefault("strides", [1,1])
        self.attr.setdefault("pads", [0,0,0,0])
        self.attr.setdefault("dilations", [1,1])

        # create pooling layer hardware
        return PoolingLayer(
            self.input_shape[2],
            self.input_shape[3],
            self.input_shape[1],
            pool_type = 'max',
            kernel_size = self.attr["kernel_shape"],
            stride = self.attr["strides"],
            pad = self.attr["pads"],
        )

class Parser:

    def __init__(self):

        # passes for onnx optimizer
        self.onnxoptimizer_passes = [
            "extract_constant_to_initializer",
            "eliminate_unused_initializer",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
            "fuse_matmul_add_bias_into_gemm",
        ]

        # minimum supported opset version
        self.opset_version =  25

        # set the backend string
        self.backend = "hls"

    def load_onnx_model(self, onnx_filepath):

        # load onnx model
        model = onnx.load(onnx_filepath)

        # validate model
        onnx.checker.check_model(model)

        # # check opset version
        # assert model.opset_import.version >= self.opset_version, f"ONNX Operator version {model.opset_import.version} not supported!"

        # remove doc strings
        onnx.helper.strip_doc_string(model)

        # add inputs from initializers
        onnx_helper.add_input_from_initializer(model) #Seems to be necessary for conv layers from pytorch (at least)

        # perform optimization passes
        model_opt = optimizer.optimize(model,
                passes=self.onnxoptimizer_passes)

        # manual opt to convert matmul to gemm
        model_opt = onnx_helper.convert_matmul_to_gemm(model_opt)

        # infer shapes of optimised model
        model_opt = onnx.shape_inference.infer_shapes(model_opt)

        # get rid of transpose and reshape to Gemm layers (but preserve shape info)
        model_opt = onnx_helper.remove_transpose_reshape(model_opt)

        # check optimized model
        onnx.checker.check_model(model_opt)

        onnx.save(model_opt, "model_opt.onnx")

        return model_opt

    def validate_onnx_model(self, onnx_model):
        pass

    def get_hardware_from_onnx_node(self, graph, node):

        # register converters
        converter = {
            LAYER_TYPE.Convolution: ParseOnnxConvNode,
            LAYER_TYPE.InnerProduct: ParseOnnxInnerProductNode,
            LAYER_TYPE.Pooling: ParseOnnxPoolingNode,
            LAYER_TYPE.ReLU: ParseOnnxReLUNode,
        }

        # get the node type
        node_type = from_onnx_op_type(node.op_type)
        # try converter
        try:
            return converter[node_type](graph, node, backend=self.backend)
        except KeyError:
            print(f"ERROR: {node_type} not supported, exiting now")
            exit()


    def onnx_to_fpgaconvnet(self, onnx_filepath):

        # load the onnx model
        onnx_model = self.load_onnx_model(onnx_filepath)
        # onnx_model = onnx.load(onnx_filepath)

        # create a networkx graph
        graph = nx.DiGraph()

        # add nodes from onnx to the graph
        for node in onnx_model.graph.node:

            # get the hardware for the node
            hardware = self.get_hardware_from_onnx_node(
                    onnx_model.graph, node)

            # add node to graph
            graph.add_node(hardware.name, **hardware.get_node_info())

        # return the graph
        return graph

    def prototxt_to_fpgaconvnet(self, proto_filepath):
        pass

    def get_layer_hardware(self, layer_type, config={}):
        pass

    def rename_graph(self):
        pass

if __name__ == "__main__":

    p = Parser()

    print("parsing alexnet")
    g = p.onnx_to_fpgaconvnet("../samo/models/alexnet.onnx")
    # g = p.onnx_to_fpgaconvnet("model_opt.onnx")

