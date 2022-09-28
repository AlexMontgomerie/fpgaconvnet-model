import onnx
import numpy as np
import importlib

from fpgaconvnet.models.layers import BatchNormLayer
from fpgaconvnet.models.layers import InnerProductLayer
from fpgaconvnet.models.layers import PoolingLayer
from fpgaconvnet.models.layers import ReLULayer

import fpgaconvnet.parser.onnx.helper as onnx_helper

from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type

class ParseOnnxNode:

    def __init__(self, graph, n, backend="hls"):

        # save node
        self.node = n

        # backend string
        self.backend = backend

        # get name of node
        self.name = onnx_helper.format_onnx_name(n)

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
        self.attr = onnx_helper.format_attr(n.attribute)

    def get_hardware(self):
        raise TypeError(f"{self.layer_type} not implemented!")

    def get_node_info(self):
        return {
            "type" : self.layer_type,
            "hw" : self.get_hardware()
        }

    def get_edges_out(self, model):
        try:
            next_node = next(filter(lambda x: x.input[0] == self.node.output[0], model.graph.node))
            return [(self.name, onnx_helper.format_onnx_name(next_node))]
        except StopIteration:
            return []

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
            self.input_shape[2] if len(self.input_shape) == 4 else 1,
            self.input_shape[3] if len(self.input_shape) == 4 else 1,
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


