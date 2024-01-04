import importlib
from dataclasses import dataclass

import numpy as np
import onnx

import fpgaconvnet.parser.onnx.helper as onnx_helper
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.layers import *
from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type


class ParseOnnxNode:

    def __init__(self, graph, n, quant_format, dimensionality=2, backend="chisel", regression_model="linear_regression", convert_gemm_to_conv=False):

        # refrence of the graph
        self.graph = graph

        # quantisation format
        self.quant_format = quant_format

        # model dimensionality
        self.dimensionality = dimensionality

        # save node
        self.node = n

        # backend string
        self.backend = backend

        # regression model
        self.regression_model = regression_model

        # get name of node
        self.name = onnx_helper.format_onnx_name(n)

        # get the layer type
        self.layer_type = from_onnx_op_type(n.op_type)

        # get inputs and outputs
        all_tensors = [ *graph.input, *graph.output, *graph.value_info, *graph.initializer ]
        self.inputs = [ next(filter(lambda x: x.name == i, all_tensors)) for i in n.input if i != ""]
        self.outputs = [ next(filter(lambda x: x.name == i, all_tensors)) for i in n.output if i != ""]

        # input and output shape
        self.input_shape = [ x.dim_value for x in self.inputs[0].type.tensor_type.shape.dim ]
        self.output_shape = [ x.dim_value for x in self.outputs[0].type.tensor_type.shape.dim ]

        # get attributes
        self.attr = onnx_helper.format_attr(n.attribute)

        # flag to convert gemm nodes to convolution
        self.convert_gemm_to_conv = convert_gemm_to_conv

        # stats for encoding weights and activations
        self.attr.setdefault("input_compression_ratio", [1.0]*len(self.inputs))
        self.attr.setdefault("output_compression_ratio", [1.0]*len(self.outputs))
        self.attr.setdefault("weight_compression_ratio", [1.0])

        # get hardware
        self.hw = self.get_hardware()

    def get_hardware(self):
        raise TypeError(f"{self.layer_type} not implemented!")

    def get_node_info(self):
        return {
            "type" : self.layer_type,
            "onnx_node": self.node.name,
            "onnx_input": list(self.node.input),
            "onnx_output": list(self.node.output),
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
                lambda x: self.node.input[0] in x.output, model.graph.node))
            return [(onnx_helper.format_onnx_name(prev_node), self.name)]
        except StopIteration:
            return []

class ParseOnnxConvNode(ParseOnnxNode):

    def get_hardware(self):

        # default attributes
        if self.dimensionality == 2:
            self.attr.setdefault("group", 1)
            self.attr.setdefault("strides", [1,1])
            self.attr.setdefault("pads", [0,0,0,0])
            self.attr.setdefault("dilations", [1,1])
            self.attr.setdefault("channel_sparsity_hist", [])
        else:
            self.attr.setdefault("group", 1)
            self.attr.setdefault("strides", [1,1,1])
            self.attr.setdefault("pads", [0,0,0,0,0,0])
            self.attr.setdefault("dilations", [1,1,1])
            self.attr.setdefault("channel_sparsity_hist", [])

        # sparsity check
        if len(self.attr["channel_sparsity_hist"]) == 0:
            type_flag = "dense"
        else:
            assert len(self.attr["channel_sparsity_hist"]) == self.input_shape[1]*(self.attr["kernel_shape"][0]*self.attr["kernel_shape"][1]+1)
            assert np.max(self.attr["channel_sparsity_hist"]) <= 1.0, "histogram values should be normalized"
            channel_sparsity_hist = np.array(self.attr["channel_sparsity_hist"]).reshape(self.input_shape[1],-1)
            channel_sparsity_avg = np.sum(channel_sparsity_hist * np.arange(0,self.attr["kernel_shape"][0]*self.attr["kernel_shape"][1]+1) / (self.attr["kernel_shape"][0]*self.attr["kernel_shape"][1]), axis=1)
            layer_sparsity_avg = np.mean(channel_sparsity_avg)
            if layer_sparsity_avg < 0.1:
                type_flag = "dense" # sparsity is too small, use dense instead
            elif self.attr["kernel_shape"][0] == 1 and self.attr["kernel_shape"][1] == 1:
                type_flag = "pointwise_sparse"
            else:
                type_flag = "sparse"

        # return hardware
        if self.dimensionality == 2:
            if type_flag == "dense":
                return ConvolutionLayer(
                    self.output_shape[1],
                    self.input_shape[2],
                    self.input_shape[3],
                    self.input_shape[1],
                    kernel_rows=self.attr["kernel_shape"][0],
                    kernel_cols=self.attr["kernel_shape"][1],
                    stride_rows=self.attr["strides"][0],
                    stride_cols=self.attr["strides"][1],
                    pad_top     = self.attr["pads"][0],
                    pad_left    = self.attr["pads"][1],
                    pad_bottom  = self.attr["pads"][2],
                    pad_right   = self.attr["pads"][3],
                    groups = self.attr["group"],
                    input_t  = FixedPoint(self.quant_format["input_t"]["width"],
                        self.quant_format["input_t"]["binary_point"]),
                    output_t = FixedPoint(self.quant_format["output_t"]["width"],
                        self.quant_format["output_t"]["binary_point"]),
                    weight_t = FixedPoint(self.quant_format["weight_t"]["width"],
                        self.quant_format["weight_t"]["binary_point"]),
                    acc_t    = FixedPoint(self.quant_format["acc_t"]["width"],
                        self.quant_format["acc_t"]["binary_point"]),
                    has_bias = len(self.inputs) == 3,
                    block_floating_point = self.quant_format["block_floating_point"],
                    backend=self.backend,
                    regression_model=self.regression_model,
                    input_compression_ratio = self.attr["input_compression_ratio"],
                    output_compression_ratio = self.attr["output_compression_ratio"],
                    weight_compression_ratio = self.attr["weight_compression_ratio"]
                )
            elif type_flag == "sparse":
                return ConvolutionSparseLayer(
                    self.output_shape[1],
                    self.input_shape[2],
                    self.input_shape[3],
                    self.input_shape[1],
                    kernel_rows=self.attr["kernel_shape"][0],
                    kernel_cols=self.attr["kernel_shape"][1],
                    stride_rows=self.attr["strides"][0],
                    stride_cols=self.attr["strides"][1],
                    pad_top     = self.attr["pads"][0],
                    pad_left    = self.attr["pads"][1],
                    pad_bottom  = self.attr["pads"][2],
                    pad_right   = self.attr["pads"][3],
                    groups = self.attr["group"],
                    input_t  = FixedPoint(self.quant_format["input_t"]["width"],
                        self.quant_format["input_t"]["binary_point"]),
                    output_t = FixedPoint(self.quant_format["output_t"]["width"],
                        self.quant_format["output_t"]["binary_point"]),
                    weight_t = FixedPoint(self.quant_format["weight_t"]["width"],
                        self.quant_format["weight_t"]["binary_point"]),
                    acc_t    = FixedPoint(self.quant_format["acc_t"]["width"],
                        self.quant_format["acc_t"]["binary_point"]),
                    has_bias = len(self.inputs) == 3,
                    channel_sparsity_hist = channel_sparsity_hist.flatten(),
                    skip_all_zero_window = True,
                    block_floating_point = self.quant_format["block_floating_point"],
                    backend=self.backend,
                    regression_model=self.regression_model,
                    input_compression_ratio = self.attr["input_compression_ratio"],
                    output_compression_ratio = self.attr["output_compression_ratio"],
                    weight_compression_ratio = self.attr["weight_compression_ratio"]
                )
            elif type_flag == "pointwise_sparse":
                return ConvolutionPointwiseSparseLayer(
                    self.output_shape[1],
                    self.input_shape[2],
                    self.input_shape[3],
                    self.input_shape[1],
                    stride_rows=self.attr["strides"][0],
                    stride_cols=self.attr["strides"][1],
                    pad_top     = self.attr["pads"][0],
                    pad_left    = self.attr["pads"][1],
                    pad_bottom  = self.attr["pads"][2],
                    pad_right   = self.attr["pads"][3],
                    groups = self.attr["group"],
                    input_t  = FixedPoint(self.quant_format["input_t"]["width"],
                        self.quant_format["input_t"]["binary_point"]),
                    output_t = FixedPoint(self.quant_format["output_t"]["width"],
                        self.quant_format["output_t"]["binary_point"]),
                    weight_t = FixedPoint(self.quant_format["weight_t"]["width"],
                        self.quant_format["weight_t"]["binary_point"]),
                    acc_t    = FixedPoint(self.quant_format["acc_t"]["width"],
                        self.quant_format["acc_t"]["binary_point"]),
                    has_bias = len(self.inputs) == 3,
                    channel_sparsity_avg = channel_sparsity_avg,
                    clusters = 1,
                    block_floating_point = self.quant_format["block_floating_point"],
                    backend=self.backend,
                    regression_model=self.regression_model,
                    input_compression_ratio = self.attr["input_compression_ratio"],
                    output_compression_ratio = self.attr["output_compression_ratio"],
                    weight_compression_ratio = self.attr["weight_compression_ratio"]
                )
        elif self.dimensionality == 3:
            return ConvolutionLayer3D(
                filters=self.output_shape[1],
                rows=self.input_shape[3],
                cols=self.input_shape[4],
                depth=self.input_shape[2],
                channels=self.input_shape[1],
                kernel_rows=self.attr["kernel_shape"][1],
                kernel_cols=self.attr["kernel_shape"][2],
                kernel_depth=self.attr["kernel_shape"][0],
                stride_rows=self.attr["strides"][1],
                stride_cols=self.attr["strides"][2],
                stride_depth=self.attr["strides"][0],
                pad_front   = self.attr["pads"][0],
                pad_top     = self.attr["pads"][1],
                pad_left    = self.attr["pads"][2],
                pad_back    = self.attr["pads"][3],
                pad_bottom  = self.attr["pads"][4],
                pad_right   = self.attr["pads"][5],
                groups = self.attr["group"],
                input_t  = FixedPoint(self.quant_format["input_t"]["width"],
                    self.quant_format["input_t"]["binary_point"]),
                output_t = FixedPoint(self.quant_format["output_t"]["width"],
                    self.quant_format["output_t"]["binary_point"]),
                weight_t = FixedPoint(self.quant_format["weight_t"]["width"],
                    self.quant_format["weight_t"]["binary_point"]),
                acc_t    = FixedPoint(self.quant_format["acc_t"]["width"],
                    self.quant_format["acc_t"]["binary_point"]),
                has_bias = len(self.inputs) == 3,
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"],
                weight_compression_ratio = self.attr["weight_compression_ratio"]
            )
        else:
            raise NotImplementedError(f"dimensionality {self.dimensionality} not supported for ConvolutionLayer")

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
        if self.dimensionality == 2:
            self.attr.setdefault("group", 1)
            self.attr.setdefault("strides", [1,1])
            self.attr.setdefault("pads", [0,0,0,0])
            self.attr.setdefault("dilations", [1,1])
        else:
            self.attr.setdefault("group", 1)
            self.attr.setdefault("strides", [1,1,1])
            self.attr.setdefault("pads", [0,0,0,0,0,0])
            self.attr.setdefault("dilations", [1,1,1])

        # return hardware
        if not self.convert_gemm_to_conv:
            if self.dimensionality == 2:
                return InnerProductLayer(
                    self.output_shape[1],
                    1, 1,
                    np.prod(self.input_shape[1:]),
                    input_t  = FixedPoint(self.quant_format["input_t"]["width"],
                        self.quant_format["input_t"]["binary_point"]),
                    output_t = FixedPoint(self.quant_format["output_t"]["width"],
                        self.quant_format["output_t"]["binary_point"]),
                    weight_t = FixedPoint(self.quant_format["weight_t"]["width"],
                        self.quant_format["weight_t"]["binary_point"]),
                    acc_t    = FixedPoint(self.quant_format["acc_t"]["width"],
                        self.quant_format["acc_t"]["binary_point"]),
                    has_bias = len(self.inputs) == 3,
                    block_floating_point = self.quant_format["block_floating_point"],
                    backend=self.backend,
                    regression_model=self.regression_model,
                    input_compression_ratio = self.attr["input_compression_ratio"],
                    output_compression_ratio = self.attr["output_compression_ratio"],
                    weight_compression_ratio = self.attr["weight_compression_ratio"]
                )
            elif self.dimensionality == 3:
                return InnerProductLayer3D(
                    self.output_shape[1],
                    1, 1, 1,
                    np.prod(self.input_shape[1:]),
                    input_t  = FixedPoint(self.quant_format["input_t"]["width"],
                        self.quant_format["input_t"]["binary_point"]),
                    output_t = FixedPoint(self.quant_format["output_t"]["width"],
                        self.quant_format["output_t"]["binary_point"]),
                    weight_t = FixedPoint(self.quant_format["weight_t"]["width"],
                        self.quant_format["weight_t"]["binary_point"]),
                    acc_t    = FixedPoint(self.quant_format["acc_t"]["width"],
                        self.quant_format["acc_t"]["binary_point"]),
                    has_bias = len(self.inputs) == 3,
                    backend=self.backend,
                    regression_model=self.regression_model,
                    input_compression_ratio = self.attr["input_compression_ratio"],
                    output_compression_ratio = self.attr["output_compression_ratio"],
                    weight_compression_ratio = self.attr["weight_compression_ratio"]
                )
            else:
                raise NotImplementedError(f"dimensionality {self.dimensionality} not supported for InnerProductLayer")
        else:
            # change the layer type
            self.layer_type = LAYER_TYPE.Convolution
            if self.dimensionality == 2:
                return ConvolutionLayer(
                    self.output_shape[1],
                    1, 1,
                    np.prod(self.input_shape[1:]),
                    kernel_rows=1, kernel_cols=1,
                    stride_rows=1, stride_cols=1,
                    pad_top=0, pad_bottom=0,
                    pad_left=0, pad_right=0,
                    groups = 1,
                    input_t  = FixedPoint(self.quant_format["input_t"]["width"],
                        self.quant_format["input_t"]["binary_point"]),
                    output_t = FixedPoint(self.quant_format["output_t"]["width"],
                        self.quant_format["output_t"]["binary_point"]),
                    weight_t = FixedPoint(self.quant_format["weight_t"]["width"],
                        self.quant_format["weight_t"]["binary_point"]),
                    acc_t    = FixedPoint(self.quant_format["acc_t"]["width"],
                        self.quant_format["acc_t"]["binary_point"]),
                    has_bias = len(self.inputs) == 3,
                    backend=self.backend,
                    regression_model=self.regression_model,
                    input_compression_ratio = self.attr["input_compression_ratio"],
                    output_compression_ratio = self.attr["output_compression_ratio"],
                    weight_compression_ratio = self.attr["weight_compression_ratio"]
                    )
            elif self.dimensionality == 3:
                return ConvolutionLayer3D(
                    self.output_shape[1],
                    1, 1, 1,
                    np.prod(self.input_shape[1:]),
                    kernel_rows=1, kernel_cols=1, kernel_depth=1,
                    stride_rows=1, stride_cols=1, stride_depth=1,
                    pad_top=0, pad_bottom=0,
                    pad_left=0, pad_right=0,
                    pad_front=0, pad_back=0,
                    groups = 1,
                    input_t  = FixedPoint(self.quant_format["input_t"]["width"],
                        self.quant_format["input_t"]["binary_point"]),
                    output_t = FixedPoint(self.quant_format["output_t"]["width"],
                        self.quant_format["output_t"]["binary_point"]),
                    weight_t = FixedPoint(self.quant_format["weight_t"]["width"],
                        self.quant_format["weight_t"]["binary_point"]),
                    acc_t    = FixedPoint(self.quant_format["acc_t"]["width"],
                        self.quant_format["acc_t"]["binary_point"]),
                    has_bias = len(self.inputs) == 3,
                    backend=self.backend,
                    regression_model=self.regression_model,
                    input_compression_ratio = self.attr["input_compression_ratio"],
                    output_compression_ratio = self.attr["output_compression_ratio"],
                    weight_compression_ratio = self.attr["weight_compression_ratio"]
                )
            else:
                raise NotImplementedError(f"dimensionality {self.dimensionality} not supported for InnerProductLayer")

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
        if self.dimensionality == 2:
            return ReLULayer(
                self.input_shape[2] if len(self.input_shape) == 4 else 1,
                self.input_shape[3] if len(self.input_shape) == 4 else 1,
                self.input_shape[1],
                data_t  = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return ReLULayer3D(
                self.input_shape[3] if len(self.input_shape) == 5 else 1,
                self.input_shape[4] if len(self.input_shape) == 5 else 1,
                self.input_shape[2] if len(self.input_shape) == 5 else 1,
                self.input_shape[1],
                data_t  = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        else:
            raise NotImplementedError(f"dimensionality {self.dimensionality} not supported for ReLULayer")



class ParseOnnxHardSwishNode(ParseOnnxNode):

    def get_hardware(self):

        # return hardware
        if self.dimensionality == 2:
            return HardswishLayer(
                self.input_shape[2] if len(self.input_shape) == 4 else 1,
                self.input_shape[3] if len(self.input_shape) == 4 else 1,
                self.input_shape[1],
                input_t = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                output_t = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return HardswishLayer3D(
                self.input_shape[3] if len(self.input_shape) == 5 else 1,
                self.input_shape[4] if len(self.input_shape) == 5 else 1,
                self.input_shape[2] if len(self.input_shape) == 5 else 1,
                self.input_shape[1],
                input_t = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                output_t = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        else:
            raise NotImplementedError(f"dimensionality {self.dimensionality} not supported for HardSwish")

class ParseOnnxChopNode(ParseOnnxNode):

    def get_hardware(self):

        # get the split data
        split = onnx.numpy_helper.to_array(next(filter(
            lambda x: x.name == self.inputs[1].name, self.graph.initializer)))

        # check right number of split values
        assert len(self.outputs) == len(split)
        assert sum(split) == self.input_shape[1]

        # return hardware
        if self.dimensionality == 2:
            return ChopLayer(
                self.input_shape[2] if len(self.input_shape) == 4 else 1,
                self.input_shape[3] if len(self.input_shape) == 4 else 1,
                self.input_shape[1],
                split,
                ports_out=len(self.outputs),
                data_t= FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"],
            )
        else:
            raise NotImplementedError(f"dimensionality {self.dimensionality} not supported for ReLULayer")


class ParseOnnxThresholdedReLUNode(ParseOnnxNode):

    def get_hardware(self):

        # return hardware
        return ThresholdedReLULayer(
            self.input_shape[2] if len(self.input_shape) == 4 else 1,
            self.input_shape[3] if len(self.input_shape) == 4 else 1,
            self.input_shape[1],
            self.attr["alpha"],
            data_t  = FixedPoint(self.quant_format["data_t"]["width"], self.quant_format["data_t"]["binary_point"]),
            input_compression_ratio = self.attr["input_compression_ratio"],
            output_compression_ratio = self.attr["output_compression_ratio"]
        )

class ParseOnnxActivationNode(ParseOnnxNode):

    def get_hardware(self):

        if self.layer_type == LAYER_TYPE.ReLU:
            activation_type = "relu"
        elif self.layer_type == LAYER_TYPE.Sigmoid:
            activation_type = "sigmoid"
        elif self.layer_type == LAYER_TYPE.HardSigmoid:
            activation_type = "hardsigmoid"
        elif self.layer_type == LAYER_TYPE.HardSwish:
            activation_type = "hardswish"
        else:
            raise Exception("Unsupported activation function: {}".format(self.layer_type))

        # return hardware
        if self.dimensionality == 2:
            # todo: Activation layer not implemented for 2D
            return ReLULayer(
                self.input_shape[2] if len(self.input_shape) == 4 else 1,
                self.input_shape[3] if len(self.input_shape) == 4 else 1,
                self.input_shape[1],
                data_t  = FixedPoint(self.quant_format["data_t"]["width"], self.quant_format["data_t"]["binary_point"]),
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return ActivationLayer3D(
                self.input_shape[3] if len(self.input_shape) == 5 else 1,
                self.input_shape[4] if len(self.input_shape) == 5 else 1,
                self.input_shape[2] if len(self.input_shape) == 5 else 1,
                self.input_shape[1], activation_type=activation_type,
                data_t  = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        else:
            raise NotImplementedError(f"dimensionality {self.dimensionality} not supported for ActivationLayer")


class ParseOnnxPoolingNode(ParseOnnxNode):

    def get_hardware(self):

        # default attributes
        self.attr.setdefault("strides", [1,1])
        self.attr.setdefault("pads", [0,0,0,0])
        self.attr.setdefault("dilations", [1,1])

        # create pooling layer hardware
        if self.dimensionality == 2:
            return PoolingLayer(
                self.input_shape[2],
                self.input_shape[3],
                self.input_shape[1],
                pool_type = 'max',
                kernel_rows = self.attr["kernel_shape"][0],
                kernel_cols = self.attr["kernel_shape"][1],
                stride_rows = self.attr["strides"][0],
                stride_cols = self.attr["strides"][1],
                pad_top     = self.attr["pads"][0],
                pad_left    = self.attr["pads"][1],
                pad_bottom  = self.attr["pads"][2],
                pad_right   = self.attr["pads"][3],
                data_t  = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return PoolingLayer3D(
                self.input_shape[3],
                self.input_shape[4],
                self.input_shape[2],
                self.input_shape[1],
                pool_type = 'max', # TODO
                kernel_rows = self.attr["kernel_shape"][1],
                kernel_cols = self.attr["kernel_shape"][2],
                kernel_depth = self.attr["kernel_shape"][0],
                stride_rows = self.attr["strides"][1],
                stride_cols = self.attr["strides"][2],
                stride_depth = self.attr["strides"][0],
                pad_front   = self.attr["pads"][0],
                pad_top     = self.attr["pads"][1],
                pad_left    = self.attr["pads"][2],
                pad_back    = self.attr["pads"][3],
                pad_bottom  = self.attr["pads"][4],
                pad_right   = self.attr["pads"][5],
                data_t  = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        else:
            raise NotImplementedError(f"dimensionality {self.dimensionality} not supported")

class ParseOnnxReSizeNode(ParseOnnxNode):

    def get_hardware(self):

        if self.dimensionality == 2:
            return ReSizeLayer(
                self.input_shape[2] if len(self.input_shape) == 4 else 1,
                self.input_shape[3] if len(self.input_shape) == 4 else 1,
                self.input_shape[1],
                scales=[1,1,2,2], # TODO: get from the model
                data_t  = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return ReSizeLayer3D(
                self.input_shape[3] if len(self.input_shape) == 5 else 1,
                self.input_shape[4] if len(self.input_shape) == 5 else 1,
                self.input_shape[2] if len(self.input_shape) == 5 else 1,
                self.input_shape[1],
                scales=[1,1,2,2,2], # TODO: get from the model
                data_t  = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"])

class ParseOnnxNOPNode(ParseOnnxNode):

    def get_hardware(self):

        print(f"CRITICAL WARNING: node {self.name} is skipped in hardware")

        # # change the layer type
        # self.layer_type = LAYER_TYPE.Squeeze

        # create pooling layer hardware
        if self.dimensionality == 2:
            return SqueezeLayer(
                self.input_shape[2] if len(self.input_shape) == 4 else 1,
                self.input_shape[3] if len(self.input_shape) == 4 else 1,
                self.input_shape[1],
                1, 1,
                data_t  = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return SqueezeLayer3D(
                self.input_shape[3] if len(self.input_shape) == 5 else 1,
                self.input_shape[4] if len(self.input_shape) == 5 else 1,
                self.input_shape[2] if len(self.input_shape) == 5 else 1,
                self.input_shape[1],
                1, 1,
                data_t  = FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )

class ParseOnnxGlobalPoolingNode(ParseOnnxNode):

    def get_hardware(self):

        # create Average pooling layer hardware
        if self.dimensionality == 2:
            return GlobalPoolingLayer(
                self.input_shape[2],
                self.input_shape[3],
                self.input_shape[1],
                data_t=FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                acc_t = FixedPoint(self.quant_format["acc_t"]["width"],
                    self.quant_format["acc_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return GlobalPoolingLayer3D(
                self.input_shape[3],
                self.input_shape[4],
                self.input_shape[2],
                self.input_shape[1],
                data_t=FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                acc_t = FixedPoint(self.quant_format["acc_t"]["width"],
                    self.quant_format["acc_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )

class ParseOnnxEltWiseNode(ParseOnnxNode):

    def get_hardware(self):

        if self.node.op_type not in ["Add", "Mul"]:
            raise TypeError(f"unsported eltwise type {self.node.op_type}")
        op_type = self.node.op_type.lower()

        # create Average pooling layer hardware
        if self.dimensionality == 2:
            return EltWiseLayer(
                self.input_shape[2],
                self.input_shape[3],
                self.input_shape[1],
                ports_in=len(self.inputs),
                op_type=op_type,
                broadcast=False, # TODO: parse from the onnx
                data_t= FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                acc_t = FixedPoint(self.quant_format["acc_t"]["width"],
                    self.quant_format["acc_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return EltWiseLayer3D(
                self.input_shape[3],
                self.input_shape[4],
                self.input_shape[2],
                self.input_shape[1],
                ports_in=len(self.inputs),
                op_type=op_type,
                broadcast=False, # TODO: parse from the onnx
                data_t= FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                acc_t = FixedPoint(self.quant_format["acc_t"]["width"],
                    self.quant_format["acc_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        else:
            raise NotImplementedError(f"dimensionality {self.dimensionality} not supported")


    def get_edges_in(self, model):
        try:
            edges = []
            prev_nodes = filter(lambda x: x.output[0] in self.node.input, model.graph.node)
            for prev_node in prev_nodes:
                edges.append((onnx_helper.format_onnx_name(prev_node), self.name))
            return edges
        except StopIteration:
            return []

class ParseOnnxConcatNode(ParseOnnxNode):

    def get_hardware(self):

        # get the shape per input
        input_shape = [ [ x.dim_value for x in \
                i.type.tensor_type.shape.dim ] for i in self.inputs ]

        # create Average pooling layer hardware
        if self.dimensionality == 2:
            return ConcatLayer(
                input_shape[0][2],
                input_shape[0][3],
                [ x[1] for x in input_shape ],
                ports_in=len(self.inputs),
                data_t= FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        elif self.dimensionality == 3:
            return ConcatLayer3D(
                input_shape[0][3],
                input_shape[0][4],
                input_shape[0][2],
                [ x[1] for x in input_shape ],
                ports_in=len(self.inputs),
                data_t= FixedPoint(self.quant_format["data_t"]["width"],
                    self.quant_format["data_t"]["binary_point"]),
                backend=self.backend,
                regression_model=self.regression_model,
                input_compression_ratio = self.attr["input_compression_ratio"],
                output_compression_ratio = self.attr["output_compression_ratio"]
            )
        else:
            raise NotImplementedError(f"dimensionality {self.dimensionality} not supported")


    def get_edges_in(self, model):
        try:
            edges = []
            prev_nodes = filter(lambda x: x.output[0] in self.node.input, model.graph.node)
            for prev_node in prev_nodes:
                edges.append((onnx_helper.format_onnx_name(prev_node), self.name))
            return edges
        except StopIteration:
            return []

