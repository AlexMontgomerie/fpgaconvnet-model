import onnx
import numpy as np
import importlib

from fpgaconvnet.models.layers import BatchNormLayer
from fpgaconvnet.models.layers import ConvolutionLayer, ConvolutionLayer3D
from fpgaconvnet.models.layers import ConvolutionSparseLayer, ConvolutionPointwiseSparseLayer
from fpgaconvnet.models.layers import InnerProductLayer, InnerProductLayer3D
from fpgaconvnet.models.layers import PoolingLayer, PoolingLayer3D
from fpgaconvnet.models.layers import ReLULayer, ReLULayer3D
from fpgaconvnet.models.layers import ThresholdedReLULayer
from fpgaconvnet.models.layers import SqueezeLayer, SqueezeLayer3D
from fpgaconvnet.models.layers import GlobalPoolingLayer, GlobalPoolingLayer3D
from fpgaconvnet.models.layers import EltWiseLayer, EltWiseLayer3D
from fpgaconvnet.models.layers import SplitLayer, SplitLayer3D
from fpgaconvnet.models.layers import ConcatLayer, ConcatLayer3D
from fpgaconvnet.models.layers import ActivationLayer3D
from fpgaconvnet.models.layers import ReSizeLayer, ReSizeLayer3D
from fpgaconvnet.models.layers import HardswishLayer, HardswishLayer3D
from fpgaconvnet.models.layers import ChopLayer

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.tools.layer_enum import LAYER_TYPE
import fpgaconvnet.tools.layer_enum as layer_enum

from google.protobuf.json_format import MessageToDict

class ParsePrototxtNode:

    def __init__(self, n, dimensionality=2, backend="chisel", regression_model="linear_regression"):

        self.dimensionality = dimensionality

        # save node
        self.node = n

        # backend string
        self.backend = backend

        # regression string
        self.regression_model = regression_model

        # get name of node
        self.name = n.name

        # get the layer type
        self.layer_type = layer_enum.from_proto_layer_type(n.type)

        # get the op type
        self.op_type = n.op_type

        # get inputs and outputs
        self.inputs = [ stream.node for stream in self.node.streams_in ]
        self.outputs = [ stream.node for stream in self.node.streams_out ]

        # input and output shape
        self.input_shape = [ n.parameters.rows_in,
                n.parameters.cols_in, n.parameters.channels_in ]
        self.output_shape = [ n.parameters.rows_out,
                n.parameters.cols_out, n.parameters.channels_out ]

        self.attr = MessageToDict(self.node.parameters, preserving_proto_field_name=True)
        # get hardware
        self.hw = self.get_hardware()

    def get_hardware(self):
        raise TypeError(f"{self.layer_type} not implemented!")

    def get_node_info(self, graph):
        return {
            "type" : self.layer_type,
            "onnx_node": self.node.onnx_node,
            "onnx_input": graph.nodes[self.node.onnx_node]["onnx_input"],
            "onnx_output":graph.nodes[self.node.onnx_node]["onnx_output"],
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

    def get_edges_in(self):
        edges = []
        for prev_node in self.inputs:
            if prev_node != self.name:
                edges.append((prev_node, self.name))
        return edges

class ParsePrototxtConvNode(ParsePrototxtNode):

    def get_hardware(self):

        # return hardware
        if self.dimensionality == 2:
            if self.node.op_type == "dense":
                return ConvolutionLayer(
                    self.attr["channels_out"],
                    self.attr["rows_in"],
                    self.attr["cols_in"],
                    self.attr["channels_in"],
                    kernel_rows =self.attr["kernel_rows"],
                    kernel_cols =self.attr["kernel_cols"],
                    stride_rows =self.attr["stride_rows"],
                    stride_cols =self.attr["stride_cols"],
                    pad_top     =self.attr["pad_top"],
                    pad_bottom  =self.attr["pad_bottom"],
                    pad_left    =self.attr["pad_left"],
                    pad_right   =self.attr["pad_right"],
                    groups      =self.attr["groups"],
                    fine        =self.attr["fine"],
                    coarse_in   =self.attr["coarse_in"],
                    coarse_out  =self.attr["coarse_out"],
                    coarse_group=self.attr["coarse_group"],
                    input_t     =FixedPoint(self.attr["input_t"]["width"], self.attr["input_t"]["binary_point"]),
                    output_t    =FixedPoint(self.attr["output_t"]["width"], self.attr["output_t"]["binary_point"]),
                    weight_t    =FixedPoint(self.attr["weight_t"]["width"], self.attr["weight_t"]["binary_point"]),
                    acc_t       =FixedPoint(self.attr["acc_t"]["width"], self.attr["acc_t"]["binary_point"]),
                    has_bias    =self.attr["has_bias"],
                    block_floating_point =self.attr["block_floating_point"],
                    backend =self.backend,
                    regression_model =self.regression_model,
                    stream_weights=self.attr["stream_weights"],
                    use_uram = self.attr["use_uram"],
                    input_compression_ratio=self.attr["input_compression_ratio"],
                    output_compression_ratio=self.attr["output_compression_ratio"],
                    weight_compression_ratio=self.attr["weight_compression_ratio"]
                )
            elif self.node.op_type == "sparse":
                return ConvolutionSparseLayer(
                    self.attr["channels_out"],
                    self.attr["rows_in"],
                    self.attr["cols_in"],
                    self.attr["channels_in"],
                    kernel_rows =self.attr["kernel_rows"],
                    kernel_cols =self.attr["kernel_cols"],
                    stride_rows =self.attr["stride_rows"],
                    stride_cols =self.attr["stride_cols"],
                    pad_top     =self.attr["pad_top"],
                    pad_bottom  =self.attr["pad_bottom"],
                    pad_left    =self.attr["pad_left"],
                    pad_right   =self.attr["pad_right"],
                    groups      =self.attr["groups"],
                    fine        =self.attr["fine"],
                    coarse_in   =self.attr["coarse_in"],
                    coarse_out  =self.attr["coarse_out"],
                    coarse_group=self.attr["coarse_group"],
                    input_t     =FixedPoint(self.attr["input_t"]["width"], self.attr["input_t"]["binary_point"]),
                    output_t    =FixedPoint(self.attr["output_t"]["width"], self.attr["output_t"]["binary_point"]),
                    weight_t    =FixedPoint(self.attr["weight_t"]["width"], self.attr["weight_t"]["binary_point"]),
                    acc_t       =FixedPoint(self.attr["acc_t"]["width"], self.attr["acc_t"]["binary_point"]),
                    block_floating_point =self.attr["block_floating_point"],
                    has_bias    =self.attr["has_bias"],
                    channel_sparsity_hist = self.attr["sparsity"],
                    skip_all_zero_window =self.attr["skip_all_zero_window"],
                    backend =self.backend,
                    regression_model =self.regression_model,
                    stream_weights=self.attr["stream_weights"],
                    use_uram = self.attr["use_uram"],
                    input_compression_ratio=self.attr["input_compression_ratio"],
                    output_compression_ratio=self.attr["output_compression_ratio"],
                    weight_compression_ratio=self.attr["weight_compression_ratio"]
                )
            elif self.node.op_type == "pointwise_sparse":
                return ConvolutionPointwiseSparseLayer(
                    self.attr["channels_out"],
                    self.attr["rows_in"],
                    self.attr["cols_in"],
                    self.attr["channels_in"],
                    stride_rows =self.attr["stride_rows"],
                    stride_cols =self.attr["stride_cols"],
                    pad_top     =self.attr["pad_top"],
                    pad_bottom  =self.attr["pad_bottom"],
                    pad_left    =self.attr["pad_left"],
                    pad_right   =self.attr["pad_right"],
                    groups      =self.attr["groups"],
                    coarse_in   =self.attr["coarse_in"],
                    coarse_out  =self.attr["coarse_out"],
                    coarse_group=self.attr["coarse_group"],
                    input_t     =FixedPoint(self.attr["input_t"]["width"], self.attr["input_t"]["binary_point"]),
                    output_t    =FixedPoint(self.attr["output_t"]["width"], self.attr["output_t"]["binary_point"]),
                    weight_t    =FixedPoint(self.attr["weight_t"]["width"], self.attr["weight_t"]["binary_point"]),
                    acc_t       =FixedPoint(self.attr["acc_t"]["width"], self.attr["acc_t"]["binary_point"]),
                    block_floating_point =self.attr["block_floating_point"],
                    has_bias    =self.attr["has_bias"],
                    channel_sparsity_avg = self.attr["sparsity"],
                    clusters =self.attr["clusters"],
                    backend =self.backend,
                    regression_model =self.regression_model,
                    stream_weights=self.attr["stream_weights"],
                    use_uram =self.attr["use_uram"],
                    input_compression_ratio=self.attr["input_compression_ratio"],
                    output_compression_ratio=self.attr["output_compression_ratio"],
                    weight_compression_ratio=self.attr["weight_compression_ratio"]
                )
        elif self.dimensionality == 3:
            return ConvolutionLayer3D(
                self.attr["channels_out"],
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                kernel_rows =self.attr["kernel_rows"],
                kernel_cols =self.attr["kernel_cols"],
                kernel_depth=self.attr["kernel_depth"],
                stride_rows =self.attr["stride_rows"],
                stride_cols =self.attr["stride_cols"],
                stride_depth=self.attr["stride_depth"],
                pad_top     =self.attr["pad_top"],
                pad_bottom  =self.attr["pad_bottom"],
                pad_left    =self.attr["pad_left"],
                pad_right   =self.attr["pad_right"],
                pad_front   =self.attr["pad_front"],
                pad_back    =self.attr["pad_back"],
                groups      =self.attr["groups"],
                fine        =self.attr["fine"],
                coarse_in   =self.attr["coarse_in"],
                coarse_out  =self.attr["coarse_out"],
                coarse_group=self.attr["coarse_group"],
                input_t     =FixedPoint(self.attr["input_t"]["width"], self.attr["input_t"]["binary_point"]),
                output_t    =FixedPoint(self.attr["output_t"]["width"], self.attr["output_t"]["binary_point"]),
                weight_t    =FixedPoint(self.attr["weight_t"]["width"], self.attr["weight_t"]["binary_point"]),
                acc_t       =FixedPoint(self.attr["acc_t"]["width"], self.attr["acc_t"]["binary_point"]),
                block_floating_point =self.attr["block_floating_point"],
                has_bias    =self.attr["has_bias"],
                backend =self.backend,
                regression_model =self.regression_model,
                stream_weights=self.attr["stream_weights"],
                use_uram =self.attr["use_uram"],
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"],
                weight_compression_ratio=self.attr["weight_compression_ratio"]
            )
        else:
            raise NotImplementedError

    def get_node_info(self, graph):
        node_info = ParsePrototxtNode.get_node_info(self, graph)
        node_info["inputs"] = {
            "weights" : self.node.weights_path,
            "bias" : self.node.bias_path
        }
        return node_info

class ParsePrototxtInnerProductNode(ParsePrototxtNode):

    def get_hardware(self):

        # return hardware
        if self.dimensionality == 2:
            return InnerProductLayer(
                self.attr["channels_out"],
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in"],
                coarse_in   =self.attr["coarse_in"],
                coarse_out  =self.attr["coarse_out"],
                input_t     =FixedPoint(self.attr["input_t"]["width"], self.attr["input_t"]["binary_point"]),
                output_t    =FixedPoint(self.attr["output_t"]["width"], self.attr["output_t"]["binary_point"]),
                weight_t    =FixedPoint(self.attr["weight_t"]["width"], self.attr["weight_t"]["binary_point"]),
                acc_t       =FixedPoint(self.attr["acc_t"]["width"], self.attr["acc_t"]["binary_point"]),
                block_floating_point =self.attr["block_floating_point"],
                has_bias    =self.attr["has_bias"],
                backend =self.backend,
                regression_model =self.regression_model,
                stream_weights=self.attr["stream_weights"],
                use_uram =self.attr["use_uram"],
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"],
                weight_compression_ratio=self.attr["weight_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return InnerProductLayer3D(
                self.attr["channels_out"],
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                coarse_in   =self.attr["coarse_in"],
                coarse_out  =self.attr["coarse_out"],
                input_t     =FixedPoint(self.attr["input_t"]["width"], self.attr["input_t"]["binary_point"]),
                output_t    =FixedPoint(self.attr["output_t"]["width"], self.attr["output_t"]["binary_point"]),
                weight_t    =FixedPoint(self.attr["weight_t"]["width"], self.attr["weight_t"]["binary_point"]),
                acc_t       =FixedPoint(self.attr["acc_t"]["width"], self.attr["acc_t"]["binary_point"]),
                block_floating_point =self.attr["block_floating_point"],
                has_bias    =self.attr["has_bias"],
                backend =self.backend,
                regression_model =self.regression_model,
                stream_weights=self.attr["stream_weights"],
                use_uram =self.attr["use_uram"],
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"],
                weight_compression_ratio=self.attr["weight_compression_ratio"]
            )
        else:
            raise NotImplementedError

    def get_node_info(self, graph):
        node_info = ParsePrototxtNode.get_node_info(self, graph)
        node_info["inputs"] = {
            "weights" : self.node.weights_path,
            "bias" : self.node.bias_path
        }
        return node_info

class ParsePrototxtReLUNode(ParsePrototxtNode):

    def get_hardware(self):

        # return hardware
        if self.dimensionality == 2:
            return ReLULayer(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in"],
                coarse=self.attr["coarse"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return ReLULayer3D(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                coarse=self.attr["coarse"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        else:
            raise NotImplementedError

class ParsePrototxtThresholdedReLUNode(ParsePrototxtNode):

    def get_hardware(self):

        # return hardware
        return ThresholdedReLULayer(
            self.attr["rows_in"],
            self.attr["cols_in"],
            self.attr["channels_in"],
            threshold = self.attr["threshold"],
            coarse=self.attr["coarse"],
            data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
            input_compression_ratio=self.attr["input_compression_ratio"],
            output_compression_ratio=self.attr["output_compression_ratio"]
        )

class ParsePrototxtPoolingNode(ParsePrototxtNode):

    def get_hardware(self):

        # create pooling layer hardware
        if self.dimensionality == 2:
            return PoolingLayer(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in"],
                pool_type   = self.op_type,
                kernel_rows =self.attr["kernel_rows"],
                kernel_cols =self.attr["kernel_cols"],
                stride_rows =self.attr["stride_rows"],
                stride_cols =self.attr["stride_cols"],
                pad_top     =self.attr["pad_top"],
                pad_bottom  =self.attr["pad_bottom"],
                pad_left    =self.attr["pad_left"],
                pad_right   =self.attr["pad_right"],
                coarse  =self.attr["coarse"],
                data_t  =FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return PoolingLayer3D(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                pool_type   = self.op_type,
                kernel_rows =self.attr["kernel_rows"],
                kernel_cols =self.attr["kernel_cols"],
                kernel_depth=self.attr["kernel_depth"],
                stride_rows =self.attr["stride_rows"],
                stride_cols =self.attr["stride_cols"],
                stride_depth=self.attr["stride_depth"],
                pad_top     =self.attr["pad_top"],
                pad_bottom  =self.attr["pad_bottom"],
                pad_left    =self.attr["pad_left"],
                pad_right   =self.attr["pad_right"],
                pad_front   =self.attr["pad_front"],
                pad_back    =self.attr["pad_back"],
                coarse  =self.attr["coarse"],
                data_t  =FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        else:
            raise NotImplementedError

class ParsePrototxtSqueezeNode(ParsePrototxtNode):

    def get_hardware(self):

        # create pooling layer hardware
        if self.dimensionality == 2:
            return SqueezeLayer(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in"],
                coarse_in   =self.attr["coarse_in"],
                coarse_out  =self.attr["coarse_out"],
                data_t      =FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return SqueezeLayer3D(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                coarse_in   =self.attr["coarse_in"],
                coarse_out  =self.attr["coarse_out"],
                data_t      =FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )

class ParsePrototxtGlobalPoolingNode(ParsePrototxtNode):

    def get_hardware(self):

        # create Average pooling layer hardware
        if self.dimensionality == 2:
            return GlobalPoolingLayer(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                acc_t=FixedPoint(self.attr["acc_t"]["width"], self.attr["acc_t"]["binary_point"]),
                op_type=self.op_type,
                backend =self.backend,
                coarse=self.attr["coarse"],
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return GlobalPoolingLayer3D(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                acc_t=FixedPoint(self.attr["acc_t"]["width"], self.attr["acc_t"]["binary_point"]),
                op_type=self.op_type,
                backend =self.backend,
                coarse=self.attr["coarse"],
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )

class ParsePrototxtEltWiseNode(ParsePrototxtNode):

    def get_hardware(self):

        # create eltwise layer hardware
        if self.dimensionality == 2:
            return EltWiseLayer(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in"],
                ports_in=self.attr["ports_in"],
                op_type=self.op_type,
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                acc_t=FixedPoint(self.attr["acc_t"]["width"], self.attr["acc_t"]["binary_point"]),
                backend =self.backend,
                coarse=self.attr["coarse"],
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return EltWiseLayer3D(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                ports_in=self.attr["ports_in"],
                op_type=self.op_type,
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                acc_t=FixedPoint(self.attr["acc_t"]["width"], self.attr["acc_t"]["binary_point"]),
                backend =self.backend,
                coarse=self.attr["coarse"],
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )

class ParsePrototxtSplitNode(ParsePrototxtNode):

    def get_hardware(self):

        # create eltwise layer hardware
        if self.dimensionality == 2:
            return SplitLayer(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in"],
                ports_out=self.attr["ports_out"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                coarse=self.attr["coarse"],
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return SplitLayer3D(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                ports_out=self.attr["ports_out"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                coarse=self.attr["coarse"],
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )

class ParsePrototxtConcatNode(ParsePrototxtNode):

    def get_hardware(self):

        # create concat layer hardware
        if self.dimensionality == 2:
            return ConcatLayer(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in_array"],
                ports_in=self.attr["ports_in"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                coarse=self.attr["coarse"],
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return ConcatLayer3D(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in_array"],
                ports_in=self.attr["ports_in"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                coarse=self.attr["coarse"],
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"],
                weight_compression_ratio=self.attr["weight_compression_ratio"]
            )

class ParsePrototxtActivationNode(ParsePrototxtNode):

    def get_hardware(self):

        if self.dimensionality == 2:
            return ReLULayer(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in"],
                coarse=self.attr["coarse"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return ActivationLayer3D(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                self.node.op_type.lower(),
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                coarse=self.attr["coarse"],
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        else:
            raise NotImplementedError

class ParsePrototxtReSizeNode(ParsePrototxtNode):

    def get_hardware(self):

        if self.dimensionality == 2:
            return ReSizeLayer(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in"],
                self.attr["scale"],
                coarse=self.attr["coarse"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return ReSizeLayer3D(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                self.attr["scale"],
                coarse=self.attr["coarse"],
                data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"],
                weight_compression_ratio=self.attr["weight_compression_ratio"]
            )

class ParsePrototxtHardSwishNode(ParsePrototxtNode):

    def get_hardware(self):

        if self.dimensionality == 2:
            return HardswishLayer(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["channels_in"],
                coarse=self.attr["coarse"],
                input_t=FixedPoint(self.attr["input_t"]["width"], self.attr["input_t"]["binary_point"]),
                output_t=FixedPoint(self.attr["output_t"]["width"], self.attr["output_t"]["binary_point"]),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return HardswishLayer3D(
                self.attr["rows_in"],
                self.attr["cols_in"],
                self.attr["depth_in"],
                self.attr["channels_in"],
                coarse=self.attr["coarse"],
                input_t=FixedPoint(self.attr["input_t"]["width"], self.attr["input_t"]["binary_point"]),
                output_t=FixedPoint(self.attr["output_t"]["width"], self.attr["output_t"]["binary_point"]),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.attr["input_compression_ratio"],
                output_compression_ratio=self.attr["output_compression_ratio"]
            )

class ParsePrototxtChopNode(ParsePrototxtNode):

    def get_hardware(self):

        return ChopLayer(
            self.attr["rows_in"],
            self.attr["cols_in"],
            self.attr["channels_in"],
            self.attr["split"],
            coarse=self.attr["coarse"],
            ports_out=self.attr["ports_out"],
            data_t=FixedPoint(self.attr["data_t"]["width"], self.attr["data_t"]["binary_point"]),
            backend =self.backend,
            regression_model =self.regression_model,
            input_compression_ratio=self.attr["input_compression_ratio"],
            output_compression_ratio=self.attr["output_compression_ratio"]
        )