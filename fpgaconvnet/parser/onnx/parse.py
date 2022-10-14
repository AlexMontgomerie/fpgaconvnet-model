import onnx
import numpy as np
import importlib

from fpgaconvnet.models.layers import BatchNormLayer
from fpgaconvnet.models.layers import InnerProductLayer
from fpgaconvnet.models.layers import PoolingLayer
from fpgaconvnet.models.layers import ReLULayer
from fpgaconvnet.models.layers import SqueezeLayer
from fpgaconvnet.models.layers import AveragePoolingLayer
from fpgaconvnet.models.layers import EltWiseLayer

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

        # get hardware
        self.hw = self.get_hardware()

    def get_hardware(self):
        raise TypeError(f"{self.layer_type} not implemented!")

    def get_node_info(self):
        return {
            "type" : self.layer_type,
            "attr" : self.attr,
            "hw" : self.hw
        }

    def apply_config_quantisation(self, config):
        if "layers" in config: # per-layer specification
            pass # TODO:
        else:
            # get the data type configuration
            width = config["data"]["width"]
            binary_point = config["data"].get("binary_point", width//2)
            # update hardware
            self.hw.data_width = width

    def apply_QDQ_quantisation(self): #TODO
        """
        Takes the LinearQuant and Linear DeQuant from the graph, and
        infers the fixed point widths from this. Might still need to
        give the widths (bare minimum)
        """
        pass

    def apply_QCDQ_quantisation(self): #TODO
        """
        same as above, but uses a clipping node aswell to get the width
        """
        pass

    def get_edges_in(self, model):
        try:
            prev_node = next(filter(
                lambda x: x.output[0] in self.node.input[0], model.graph.node))
            return [(onnx_helper.format_onnx_name(prev_node), self.name)]
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

    def get_node_info(self):
        node_info = ParseOnnxNode.get_node_info(self)
        node_info["inputs"] = {
            "weights" : self.node.input[1],
            "bias" : "" }
        if len(self.node.input) == 3:
            node_info["inputs"]["bias"] = self.node.input[2]
            node_info["hw"].has_bias = True
        return node_info

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

    def get_node_info(self):
        node_info = ParseOnnxNode.get_node_info(self)
        node_info["inputs"] = {
            "weights" : self.node.input[1],
            "bias" : "" }
        if len(self.node.input) == 3:
            node_info["inputs"]["bias"] = self.node.input[2]
            node_info["hw"].has_bias = True
        return node_info

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

class ParseOnnxNOPNode(ParseOnnxNode):

    def get_hardware(self):


        print(f"CRITICAL WARNING: node {self.name} is skipped in hardware")

        # create pooling layer hardware
        return SqueezeLayer(
            self.input_shape[2] if len(self.input_shape) == 4 else 1,
            self.input_shape[3] if len(self.input_shape) == 4 else 1,
            self.input_shape[1],
            1, 1
        )

class ParseOnnxAveragePoolingNode(ParseOnnxNode):

    def get_hardware(self):

        # create Average pooling layer hardware
        return AveragePoolingLayer(
            self.input_shape[2],
            self.input_shape[3],
            self.input_shape[1]
        )

class ParseOnnxEltWiseNode(ParseOnnxNode):

    def get_hardware(self):

        op_type = None
        if self.node.op_type == "Add":
            op_type = "sum"
        elif self.node.op_type == "Mul":
            op_type = "mul"
        else:
            raise TypeError(f"unsported eltwise type {self.node.op_type}")

        # create Average pooling layer hardware
        return EltWiseLayer(
            self.input_shape[2],
            self.input_shape[3],
            self.input_shape[1],
            ports_in=len(self.inputs),
            op_type=op_type
        )

    def get_edges_in(self, model):
        try:
            edges = []
            prev_nodes = filter(lambda x: x.output[0] in self.node.input, model.graph.node)
            for prev_node in prev_nodes:
                edges.append((onnx_helper.format_onnx_name(prev_node), self.name))
            return edges
        except StopIteration:
            return []

