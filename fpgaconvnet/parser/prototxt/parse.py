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
from fpgaconvnet.models.layers import SplitLayer

import fpgaconvnet.parser.onnx.helper as onnx_helper

from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type
import fpgaconvnet.tools.layer_enum as layer_enum

class ParsePrototxtNode:

    def __init__(self, n, backend="hls"):

        # save node
        self.node = n

        # backend string
        self.backend = backend

        # get name of node
        self.name = n.name

        # get the layer type
        self.layer_type = layer_enum.from_proto_layer_type(n.type)

        # get inputs and outputs
        self.inputs = [ stream.node for stream in self.node.streams_in ]
        self.outputs = [ stream.node for stream in self.node.streams_out ]

        # input and output shape
        self.input_shape = [ n.parameters.rows_in,
                n.parameters.cols_in, n.parameters.channels_in ]
        self.output_shape = [ n.parameters.rows_out,
                n.parameters.cols_out, n.parameters.channels_out ]

        # get hardware
        self.hw = self.get_hardware()

    def get_hardware(self):
        raise TypeError(f"{self.layer_type} not implemented!")

    def get_node_info(self):
        return {
            "type" : self.layer_type,
            "onnx_node" : self.node.onnx_node,
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

    def get_edges_in(self):
        edges = []
        for prev_node in self.inputs:
            if prev_node != self.name:
                edges.append((prev_node, self.name))
        return edges

class ParsePrototxtConvNode(ParsePrototxtNode):

    def get_hardware(self):

        # import layers
        convolution = importlib.import_module(
                f"fpgaconvnet.models.layers.{self.backend}")

        # return hardware
        return convolution.ConvolutionLayer(
            self.node.parameters.channels_out,
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in,
            kernel_size =list(self.node.parameters.kernel_size),
            stride      =list(self.node.parameters.stride),
            pad         = [
                self.node.parameters.pad_top,
                self.node.parameters.pad_right,
                self.node.parameters.pad_bottom,
                self.node.parameters.pad_left],
            groups      =self.node.parameters.groups,
            fine        =self.node.parameters.fine,
            coarse_in   =self.node.parameters.coarse_in,
            coarse_out  =self.node.parameters.coarse_out,
            coarse_group=self.node.parameters.coarse_group,
            has_bias    =self.node.parameters.has_bias
        )

    def get_node_info(self):
        node_info = ParsePrototxtNode.get_node_info(self)
        node_info["inputs"] = {
            "weights" : self.node.weights_path,
            "bias" : self.node.bias_path
        }
        return node_info

class ParsePrototxtInnerProductNode(ParsePrototxtNode):

    def get_hardware(self):

        # return hardware
        return InnerProductLayer(
            self.node.parameters.channels_out,
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in,
            coarse_in   =self.node.parameters.coarse_in,
            coarse_out  =self.node.parameters.coarse_out,
            has_bias    =self.node.parameters.has_bias
        )

    def get_node_info(self):
        node_info = ParsePrototxtNode.get_node_info(self)
        node_info["inputs"] = {
            "weights" : self.node.weights_path,
            "bias" : self.node.bias_path
        }
        return node_info

class ParsePrototxtReLUNode(ParsePrototxtNode):

    def get_hardware(self):

        # return hardware
        return ReLULayer(
            self.input_shape[2] if len(self.input_shape) == 4 else 1,
            self.input_shape[3] if len(self.input_shape) == 4 else 1,
            self.input_shape[1],
        )

class ParsePrototxtPoolingNode(ParsePrototxtNode):

    def get_hardware(self):

        # create pooling layer hardware
        return PoolingLayer(
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in,
            pool_type   = 'max',
            kernel_size =list(self.node.parameters.kernel_size),
            stride      =list(self.node.parameters.stride),
            pad         = [
                self.node.parameters.pad_top,
                self.node.parameters.pad_right,
                self.node.parameters.pad_bottom,
                self.node.parameters.pad_left],
            coarse  =self.node.parameters.coarse
        )

class ParsePrototxtSqueezeNode(ParsePrototxtNode):

    def get_hardware(self):

        # create pooling layer hardware
        return SqueezeLayer(
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in,
            coarse_in   =self.node.parameters.coarse_in,
            coarse_out  =self.node.parameters.coarse_out
        )

class ParsePrototxtAveragePoolingNode(ParsePrototxtNode):

    def get_hardware(self):

        # create Average pooling layer hardware
        return AveragePoolingLayer(
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in,
        )

class ParsePrototxtEltWiseNode(ParsePrototxtNode):

    def get_hardware(self):

        # create eltwise layer hardware
        return EltWiseLayer(
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in,
            ports_in=self.node.parameters.ports_in,
            op_type="sum" # TODO
        )

class ParsePrototxtSplitNode(ParsePrototxtNode):

    def get_hardware(self):

        # create eltwise layer hardware
        return SplitLayer(
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in,
            ports_out=self.node.parameters.ports_out,
        )


