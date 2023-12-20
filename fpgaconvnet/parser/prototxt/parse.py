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
from fpgaconvnet.models.layers import ConcatLayer
from fpgaconvnet.models.layers import ActivationLayer3D
from fpgaconvnet.models.layers import ReSizeLayer
from fpgaconvnet.models.layers import HardswishLayer, HardswishLayer3D
from fpgaconvnet.models.layers import ChopLayer

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.tools.layer_enum import LAYER_TYPE
import fpgaconvnet.tools.layer_enum as layer_enum

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

        self.attr = self.node.parameters
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
                    self.node.parameters.channels_out,
                    self.node.parameters.rows_in,
                    self.node.parameters.cols_in,
                    self.node.parameters.channels_in,
                    kernel_rows =self.node.parameters.kernel_rows,
                    kernel_cols =self.node.parameters.kernel_cols,
                    stride_rows =self.node.parameters.stride_rows,
                    stride_cols =self.node.parameters.stride_cols,
                    pad_top     =self.node.parameters.pad_top,
                    pad_bottom  =self.node.parameters.pad_bottom,
                    pad_left    =self.node.parameters.pad_left,
                    pad_right   =self.node.parameters.pad_right,
                    groups      =self.node.parameters.groups,
                    fine        =self.node.parameters.fine,
                    coarse_in   =self.node.parameters.coarse_in,
                    coarse_out  =self.node.parameters.coarse_out,
                    coarse_group=self.node.parameters.coarse_group,
                    input_t     =FixedPoint(self.node.parameters.input_t.width, self.node.parameters.input_t.binary_point),
                    output_t    =FixedPoint(self.node.parameters.output_t.width, self.node.parameters.output_t.binary_point),
                    weight_t    =FixedPoint(self.node.parameters.weight_t.width, self.node.parameters.weight_t.binary_point),
                    acc_t       =FixedPoint(self.node.parameters.acc_t.width, self.node.parameters.acc_t.binary_point),
                    has_bias    =self.node.parameters.has_bias,
                    block_floating_point =self.node.parameters.block_floating_point,
                    backend =self.backend,
                    regression_model =self.regression_model,
                    stream_weights=self.node.parameters.stream_weights,
                    use_uram = self.node.parameters.use_uram,
                    input_compression_ratio=self.node.parameters.input_compression_ratio,
                    output_compression_ratio=self.node.parameters.output_compression_ratio,
                    weight_compression_ratio=self.node.parameters.weight_compression_ratio
                )
            elif self.node.op_type == "sparse":
                return ConvolutionSparseLayer(
                    self.node.parameters.channels_out,
                    self.node.parameters.rows_in,
                    self.node.parameters.cols_in,
                    self.node.parameters.channels_in,
                    kernel_rows =self.node.parameters.kernel_rows,
                    kernel_cols =self.node.parameters.kernel_cols,
                    stride_rows =self.node.parameters.stride_rows,
                    stride_cols =self.node.parameters.stride_cols,
                    pad_top     =self.node.parameters.pad_top,
                    pad_bottom  =self.node.parameters.pad_bottom,
                    pad_left    =self.node.parameters.pad_left,
                    pad_right   =self.node.parameters.pad_right,
                    groups      =self.node.parameters.groups,
                    fine        =self.node.parameters.fine,
                    coarse_in   =self.node.parameters.coarse_in,
                    coarse_out  =self.node.parameters.coarse_out,
                    coarse_group=self.node.parameters.coarse_group,
                    input_t     =FixedPoint(self.node.parameters.input_t.width, self.node.parameters.input_t.binary_point),
                    output_t    =FixedPoint(self.node.parameters.output_t.width, self.node.parameters.output_t.binary_point),
                    weight_t    =FixedPoint(self.node.parameters.weight_t.width, self.node.parameters.weight_t.binary_point),
                    acc_t       =FixedPoint(self.node.parameters.acc_t.width, self.node.parameters.acc_t.binary_point),
                    block_floating_point =self.node.parameters.block_floating_point,
                    has_bias    =self.node.parameters.has_bias,
                    channel_sparsity_hist = self.node.parameters.sparsity,
                    skip_all_zero_window =self.node.parameters.skip_all_zero_window,
                    backend =self.backend,
                    regression_model =self.regression_model,
                    stream_weights=self.node.parameters.stream_weights,
                    use_uram = self.node.parameters.use_uram,
                    input_compression_ratio=self.node.parameters.input_compression_ratio,
                    output_compression_ratio=self.node.parameters.output_compression_ratio,
                    weight_compression_ratio=self.node.parameters.weight_compression_ratio
                )
            elif self.node.op_type == "pointwise_sparse":
                return ConvolutionPointwiseSparseLayer(
                    self.node.parameters.channels_out,
                    self.node.parameters.rows_in,
                    self.node.parameters.cols_in,
                    self.node.parameters.channels_in,
                    stride_rows =self.node.parameters.stride_rows,
                    stride_cols =self.node.parameters.stride_cols,
                    pad_top     =self.node.parameters.pad_top,
                    pad_bottom  =self.node.parameters.pad_bottom,
                    pad_left    =self.node.parameters.pad_left,
                    pad_right   =self.node.parameters.pad_right,
                    groups      =self.node.parameters.groups,
                    coarse_in   =self.node.parameters.coarse_in,
                    coarse_out  =self.node.parameters.coarse_out,
                    coarse_group=self.node.parameters.coarse_group,
                    input_t     =FixedPoint(self.node.parameters.input_t.width, self.node.parameters.input_t.binary_point),
                    output_t    =FixedPoint(self.node.parameters.output_t.width, self.node.parameters.output_t.binary_point),
                    weight_t    =FixedPoint(self.node.parameters.weight_t.width, self.node.parameters.weight_t.binary_point),
                    acc_t       =FixedPoint(self.node.parameters.acc_t.width, self.node.parameters.acc_t.binary_point),
                    block_floating_point =self.node.parameters.block_floating_point,
                    has_bias    =self.node.parameters.has_bias,
                    channel_sparsity_avg = self.node.parameters.sparsity,
                    clusters =self.node.parameters.clusters,
                    backend =self.backend,
                    regression_model =self.regression_model,
                    stream_weights=self.node.parameters.stream_weights,
                    use_uram =self.node.parameters.use_uram,
                    input_compression_ratio=self.node.parameters.input_compression_ratio,
                    output_compression_ratio=self.node.parameters.output_compression_ratio,
                    weight_compression_ratio=self.node.parameters.weight_compression_ratio
                )
        elif self.dimensionality == 3:
            return ConvolutionLayer3D(
                self.node.parameters.channels_out,
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.depth_in,
                self.node.parameters.channels_in,
                kernel_rows =self.node.parameters.kernel_rows,
                kernel_cols =self.node.parameters.kernel_cols,
                kernel_depth=self.node.parameters.kernel_depth,
                stride_rows =self.node.parameters.stride_rows,
                stride_cols =self.node.parameters.stride_cols,
                stride_depth=self.node.parameters.stride_depth,
                pad_top     =self.node.parameters.pad_top,
                pad_bottom  =self.node.parameters.pad_bottom,
                pad_left    =self.node.parameters.pad_left,
                pad_right   =self.node.parameters.pad_right,
                pad_front   =self.node.parameters.pad_front,
                pad_back    =self.node.parameters.pad_back,
                groups      =self.node.parameters.groups,
                fine        =self.node.parameters.fine,
                coarse_in   =self.node.parameters.coarse_in,
                coarse_out  =self.node.parameters.coarse_out,
                coarse_group=self.node.parameters.coarse_group,
                input_t     =FixedPoint(self.node.parameters.input_t.width, self.node.parameters.input_t.binary_point),
                output_t    =FixedPoint(self.node.parameters.output_t.width, self.node.parameters.output_t.binary_point),
                weight_t    =FixedPoint(self.node.parameters.weight_t.width, self.node.parameters.weight_t.binary_point),
                acc_t       =FixedPoint(self.node.parameters.acc_t.width, self.node.parameters.acc_t.binary_point),
                block_floating_point =self.node.parameters.block_floating_point,
                has_bias    =self.node.parameters.has_bias,
                backend =self.backend,
                regression_model =self.regression_model,
                stream_weights=self.node.parameters.stream_weights,
                use_uram =self.node.parameters.use_uram,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio,
                weight_compression_ratio=self.node.parameters.weight_compression_ratio
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
                self.node.parameters.channels_out,
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.channels_in,
                coarse_in   =self.node.parameters.coarse_in,
                coarse_out  =self.node.parameters.coarse_out,
                input_t     =FixedPoint(self.node.parameters.input_t.width, self.node.parameters.input_t.binary_point),
                output_t    =FixedPoint(self.node.parameters.output_t.width, self.node.parameters.output_t.binary_point),
                weight_t    =FixedPoint(self.node.parameters.weight_t.width, self.node.parameters.weight_t.binary_point),
                acc_t       =FixedPoint(self.node.parameters.acc_t.width, self.node.parameters.acc_t.binary_point),
                block_floating_point =self.node.parameters.block_floating_point,
                has_bias    =self.node.parameters.has_bias,
                backend =self.backend,
                regression_model =self.regression_model,
                stream_weights=self.node.parameters.stream_weights,
                use_uram =self.node.parameters.use_uram,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio,
                weight_compression_ratio=self.node.parameters.weight_compression_ratio
            )
        elif self.dimensionality == 3:
            return InnerProductLayer3D(
                self.node.parameters.channels_out,
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.depth_in,
                self.node.parameters.channels_in,
                coarse_in   =self.node.parameters.coarse_in,
                coarse_out  =self.node.parameters.coarse_out,
                input_t     =FixedPoint(self.node.parameters.input_t.width, self.node.parameters.input_t.binary_point),
                output_t    =FixedPoint(self.node.parameters.output_t.width, self.node.parameters.output_t.binary_point),
                weight_t    =FixedPoint(self.node.parameters.weight_t.width, self.node.parameters.weight_t.binary_point),
                acc_t       =FixedPoint(self.node.parameters.acc_t.width, self.node.parameters.acc_t.binary_point),
                block_floating_point =self.node.parameters.block_floating_point,
                has_bias    =self.node.parameters.has_bias,
                backend =self.backend,
                regression_model =self.regression_model,
                stream_weights=self.node.parameters.stream_weights,
                use_uram =self.node.parameters.use_uram,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio,
                weight_compression_ratio=self.node.parameters.weight_compression_ratio
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
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.channels_in,
                coarse=self.node.parameters.coarse,
                data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        elif self.dimensionality == 3:
            return ReLULayer3D(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.depth_in,
                self.node.parameters.channels_in,
                coarse=self.node.parameters.coarse,
                data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        else:
            raise NotImplementedError

class ParsePrototxtThresholdedReLUNode(ParsePrototxtNode):

    def get_hardware(self):

        # return hardware
        return ThresholdedReLULayer(
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in,
            threshold = self.node.parameters.threshold,
            coarse=self.node.parameters.coarse,
            data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
            input_compression_ratio=self.node.parameters.input_compression_ratio,
            output_compression_ratio=self.node.parameters.output_compression_ratio
        )

class ParsePrototxtPoolingNode(ParsePrototxtNode):

    def get_hardware(self):

        # create pooling layer hardware
        if self.dimensionality == 2:
            return PoolingLayer(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.channels_in,
                pool_type   = self.op_type,
                kernel_rows =self.node.parameters.kernel_rows,
                kernel_cols =self.node.parameters.kernel_cols,
                stride_rows =self.node.parameters.stride_rows,
                stride_cols =self.node.parameters.stride_cols,
                pad_top     =self.node.parameters.pad_top,
                pad_bottom  =self.node.parameters.pad_bottom,
                pad_left    =self.node.parameters.pad_left,
                pad_right   =self.node.parameters.pad_right,
                coarse  =self.node.parameters.coarse,
                data_t  =FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        elif self.dimensionality == 3:
            return PoolingLayer3D(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.depth_in,
                self.node.parameters.channels_in,
                pool_type   = self.op_type,
                kernel_rows =self.node.parameters.kernel_rows,
                kernel_cols =self.node.parameters.kernel_cols,
                kernel_depth=self.node.parameters.kernel_depth,
                stride_rows =self.node.parameters.stride_rows,
                stride_cols =self.node.parameters.stride_cols,
                stride_depth=self.node.parameters.stride_depth,
                pad_top     =self.node.parameters.pad_top,
                pad_bottom  =self.node.parameters.pad_bottom,
                pad_left    =self.node.parameters.pad_left,
                pad_right   =self.node.parameters.pad_right,
                pad_front   =self.node.parameters.pad_front,
                pad_back    =self.node.parameters.pad_back,
                coarse  =self.node.parameters.coarse,
                data_t  =FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        else:
            raise NotImplementedError

class ParsePrototxtSqueezeNode(ParsePrototxtNode):

    def get_hardware(self):

        # create pooling layer hardware
        if self.dimensionality == 2:
            return SqueezeLayer(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.channels_in,
                coarse_in   =self.node.parameters.coarse_in,
                coarse_out  =self.node.parameters.coarse_out,
                data_t      =FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        elif self.dimensionality == 3:
            return SqueezeLayer3D(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.depth_in,
                self.node.parameters.channels_in,
                coarse_in   =self.node.parameters.coarse_in,
                coarse_out  =self.node.parameters.coarse_out,
                data_t      =FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )

class ParsePrototxtGlobalPoolingNode(ParsePrototxtNode):

    def get_hardware(self):

        # create Average pooling layer hardware
        if self.dimensionality == 2:
            return GlobalPoolingLayer(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.channels_in,
                data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                acc_t=FixedPoint(self.node.parameters.acc_t.width, self.node.parameters.acc_t.binary_point),
                op_type=self.op_type,
                backend =self.backend,
                coarse=self.node.parameters.coarse,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        elif self.dimensionality == 3:
            return GlobalPoolingLayer3D(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.depth_in,
                self.node.parameters.channels_in,
                data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                acc_t=FixedPoint(self.node.parameters.acc_t.width, self.node.parameters.acc_t.binary_point),
                op_type=self.op_type,
                backend =self.backend,
                coarse=self.node.parameters.coarse,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )

class ParsePrototxtEltWiseNode(ParsePrototxtNode):

    def get_hardware(self):

        # create eltwise layer hardware
        if self.dimensionality == 2:
            return EltWiseLayer(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.channels_in,
                ports_in=self.node.parameters.ports_in,
                op_type=self.op_type,
                data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                acc_t=FixedPoint(self.node.parameters.acc_t.width, self.node.parameters.acc_t.binary_point),
                backend =self.backend,
                coarse=self.node.parameters.coarse,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        elif self.dimensionality == 3:
            return EltWiseLayer3D(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.depth_in,
                self.node.parameters.channels_in,
                ports_in=self.node.parameters.ports_in,
                op_type=self.op_type,
                data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                acc_t=FixedPoint(self.node.parameters.acc_t.width, self.node.parameters.acc_t.binary_point),
                backend =self.backend,
                coarse=self.node.parameters.coarse,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )

class ParsePrototxtSplitNode(ParsePrototxtNode):

    def get_hardware(self):

        # create eltwise layer hardware
        if self.dimensionality == 2:
            return SplitLayer(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.channels_in,
                ports_out=self.node.parameters.ports_out,
                data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                coarse=self.node.parameters.coarse,
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        elif self.dimensionality == 3:
            return SplitLayer3D(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.depth_in,
                self.node.parameters.channels_in,
                ports_out=self.node.parameters.ports_out,
                data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                coarse=self.node.parameters.coarse,
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )

class ParsePrototxtConcatNode(ParsePrototxtNode):

    def get_hardware(self):

        # create eltwise layer hardware
        return ConcatLayer(
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in_array,
            ports_in=self.node.parameters.ports_in,
            data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
            coarse=self.node.parameters.coarse,
            backend =self.backend,
            regression_model =self.regression_model,
            input_compression_ratio=self.node.parameters.input_compression_ratio,
            output_compression_ratio=self.node.parameters.output_compression_ratio
        )

class ParsePrototxtActivationNode(ParsePrototxtNode):

    def get_hardware(self):

        if self.dimensionality == 2:
            return ReLULayer(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.channels_in,
                coarse=self.node.parameters.coarse,
                data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        elif self.dimensionality == 3:
            return ActivationLayer3D(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.depth_in,
                self.node.parameters.channels_in,
                self.node.op_type.lower(),
                data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
                coarse=self.node.parameters.coarse,
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        else:
            raise NotImplementedError

class ParsePrototxtReSizeNode(ParsePrototxtNode):

    def get_hardware(self):

        return ReSizeLayer(
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in,
            self.node.parameters.scale,
            coarse=self.node.parameters.coarse,
            data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
            backend =self.backend,
            regression_model =self.regression_model,
            input_compression_ratio=self.node.parameters.input_compression_ratio,
            output_compression_ratio=self.node.parameters.output_compression_ratio
        )

class ParsePrototxtHardSwishNode(ParsePrototxtNode):

    def get_hardware(self):

        if self.dimensionality == 2:
            return HardswishLayer(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.channels_in,
                coarse=self.node.parameters.coarse,
                input_t=FixedPoint(self.node.parameters.input_t.width, self.node.parameters.input_t.binary_point),
                output_t=FixedPoint(self.node.parameters.output_t.width, self.node.parameters.output_t.binary_point),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )
        elif self.dimensionality == 3:
            return HardswishLayer3D(
                self.node.parameters.rows_in,
                self.node.parameters.cols_in,
                self.node.parameters.depth_in,
                self.node.parameters.channels_in,
                coarse=self.node.parameters.coarse,
                input_t=FixedPoint(self.node.parameters.input_t.width, self.node.parameters.input_t.binary_point),
                output_t=FixedPoint(self.node.parameters.output_t.width, self.node.parameters.output_t.binary_point),
                backend =self.backend,
                regression_model =self.regression_model,
                input_compression_ratio=self.node.parameters.input_compression_ratio,
                output_compression_ratio=self.node.parameters.output_compression_ratio
            )

class ParsePrototxtChopNode(ParsePrototxtNode):

    def get_hardware(self):

        return ChopLayer(
            self.node.parameters.rows_in,
            self.node.parameters.cols_in,
            self.node.parameters.channels_in,
            self.node.parameters.split,
            coarse=self.node.parameters.coarse,
            ports_out=self.node.parameters.ports_out,
            data_t=FixedPoint(self.node.parameters.data_t.width, self.node.parameters.data_t.binary_point),
            backend =self.backend,
            regression_model =self.regression_model,
            input_compression_ratio=self.node.parameters.input_compression_ratio,
            output_compression_ratio=self.node.parameters.output_compression_ratio
        )