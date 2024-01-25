import importlib
from dataclasses import dataclass

import numpy as np
import onnx

import fpgaconvnet.parser.onnx.helper as onnx_helper
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

class ParseOnnxNode:

    def __init__(self, graph, n, quant_format, dimensionality=DIMENSIONALITY.TWO,
                 backend=BACKEND.CHISEL, convert_gemm_to_conv=False):

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

        # get name of node
        self.name = onnx_helper.format_onnx_name(n)

        # get the layer type
        self.layer_type = from_onnx_op_type(n.op_type)

        # get inputs and outputs
        all_tensors = [ *graph.input, *graph.output, *graph.value_info, *graph.initializer ]
        self.inputs = [ next(filter(lambda x: x.name == i, all_tensors)) for i in n.input ]
        self.outputs = [ next(filter(lambda x: x.name == i, all_tensors)) for i in n.output]

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

    def get_config(self):

        # initialise dictionary for args
        config = {}

        # add the spatial dimensions
        config["channels"] = self.input_shape[1]
        if len(self.input_shape) == 4 and \
                self.dimensionality == DIMENSIONALITY.TWO:
            config["rows"] = self.input_shape[2]
            config["cols"] = self.input_shape[3]
        elif len(self.input_shape) == 5 and \
                self.dimensionality == DIMENSIONALITY.THREE:
            config["depth"] = self.input_shape[2]
            config["rows"] = self.input_shape[3]
            config["cols"] = self.input_shape[4]

        # add quantisation information
        config.update(self.quant_format)

        # iterate over the attributes
        for k,v in self.attr.items():
            match k:
                case "kernel_shape":
                    config["kernel_rows"] = v[0]
                    config["kernel_cols"] = v[1]
                    if self.dimensionality == DIMENSIONALITY.THREE:
                        config["kernel_depth"] = v[2]
                case "strides":
                    config["stride_rows"] = v[0]
                    config["stride_cols"] = v[1]
                    if self.dimensionality == DIMENSIONALITY.THREE:
                        config["stride_depth"] = v[2]
                case "pads":
                    if self.dimensionality == DIMENSIONALITY.TWO:
                        config["pad_top"] = v[0]
                        config["pad_left"] = v[1]
                        config["pad_bottom"] = v[2]
                        config["pad_right"] = v[3]
                    elif self.dimensionality == DIMENSIONALITY.THREE:
                        config["pad_front"] = v[0]
                        config["pad_top"] = v[1]
                        config["pad_left"] = v[2]
                        config["pad_back"] = v[3]
                        config["pad_bottom"] = v[4]
                        config["pad_right"] = v[5]
                case _:
                    config[k] = v

        # return the config
        return config

class ParseOnnxConvNode(ParseOnnxNode):

    def get_hardware(self):

        # default attributes
        self.attr.setdefault("group", 1)
        self.attr.setdefault("strides", [1]*int(self.dimensionality))
        self.attr.setdefault("pads", [0]*2*int(self.dimensionality))
        self.attr.setdefault("dilations", [1]*int(self.dimensionality))
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

        # collect the config from the attributes
        config = self.get_config()

        # add the filters
        config["filters"] = self.output_shape[1]

        # initialise layer
        return LayerBase.build("convolution", config, self.backend, self.dimensionality) # TODO: support the sparsity hardware

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
        self.attr.setdefault("strides", [1]*int(self.dimensionality))
        self.attr.setdefault("pads", [0]*2*int(self.dimensionality))
        self.attr.setdefault("dilations", [1]*int(self.dimensionality))
        self.attr.setdefault("channel_sparsity_hist", [])

        # collect the config from the attributes
        config = self.get_config()

        # add the filters
        config["filters"] = self.output_shape[1]

        # initialise layer
        if not self.convert_gemm_to_conv:
            return LayerBase.build("inner_product", config, self.backend, self.dimensionality) # TODO: support the sparsity hardware
        else:
            return LayerBase.build("convolution", config, self.backend, self.dimensionality) # TODO: support the sparsity hardware

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

        # collect the config from the attributes
        config = self.get_config()

        # initialise layer
        return LayerBase.build("relu", config, self.backend, self.dimensionality)

class ParseOnnxHardSwishNode(ParseOnnxNode):

    def get_hardware(self):

        # collect the config from the attributes
        config = self.get_config()

        # initialise layer
        return LayerBase.build("hardswish", config, self.backend, self.dimensionality)

class ParseOnnxChopNode(ParseOnnxNode):

    def get_hardware(self):

        # get the split data
        split = onnx.numpy_helper.to_array(next(filter(
            lambda x: x.name == self.inputs[1].name, self.graph.initializer)))

        # check right number of split values
        assert len(self.outputs) == len(split)
        assert sum(split) == self.input_shape[1]

        # collect the config from the attributes
        config = self.get_config()

        # add split and ports out info
        config["split"] = split
        config["ports_out"] = len(self.outputs)

        # initialise layer
        return LayerBase.build("chop", config, self.backend, self.dimensionality)


class ParseOnnxThresholdedReLUNode(ParseOnnxNode):

    def get_hardware(self):

        # collect the config from the attributes
        config = self.get_config()

        # initialise layer
        return LayerBase.build("hardswish", config, self.backend, self.dimensionality)

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

        # collect the config from the attributes
        config = self.get_config()

        # initialise layer
        return LayerBase.build(activation_type, config, self.backend, self.dimensionality)

class ParseOnnxPoolingNode(ParseOnnxNode):

    def get_hardware(self):

        # default attributes
        self.attr.setdefault("strides", [1,1])
        self.attr.setdefault("pads", [0,0,0,0])
        self.attr.setdefault("dilations", [1,1])

        # default attributes
        self.attr.setdefault("strides", [1]*int(self.dimensionality))
        self.attr.setdefault("pads", [0]*2*int(self.dimensionality))
        self.attr.setdefault("dilations", [1]*int(self.dimensionality))

        # collect the config from the attributes
        config = self.get_config()

        # add the pool type
        config["pool_type"] = "max"

        # initialise layer
        return LayerBase.build("pooling", config, self.backend, self.dimensionality)

class ParseOnnxReSizeNode(ParseOnnxNode):

    def get_hardware(self):

        # collect the config from the attributes
        config = self.get_config()

        # add the scales
        # config["scales"] = self.attr["scales"]
        config["scales"] = [2,2,1] # TODO: get from the model

        # initialise layer
        return LayerBase.build("resize", config, self.backend, self.dimensionality)

class ParseOnnxNOPNode(ParseOnnxNode):

    def get_hardware(self):

        print(f"CRITICAL WARNING: node {self.name} is skipped in hardware")

        # collect the config from the attributes
        config = self.get_config()

        # add the coarse factors
        config["coarse_in"] = 1
        config["coarse_out"] = 1

        # initialise layer
        return LayerBase.build("squeeze", config, self.backend, self.dimensionality)

class ParseOnnxGlobalPoolingNode(ParseOnnxNode):

    def get_hardware(self):

        # collect the config from the attributes
        config = self.get_config()
        print(config)

        # initialise layer
        return LayerBase.build("global_pool", config, self.backend, self.dimensionality)

class ParseOnnxEltWiseNode(ParseOnnxNode):

    def get_hardware(self):

        if self.node.op_type not in ["Add", "Mul"]:
            raise TypeError(f"unsported eltwise type {self.node.op_type}")
        op_type = self.node.op_type.lower()

        # collect the config from the attributes
        config = self.get_config()

        # add the additional parameters
        config["op_type"] = op_type
        config["ports"] = len(self.inputs)
        config["broadcast"] = False # TODO: parse from the onnx

        # initialise layer
        return LayerBase.build("eltwise", config, self.backend, self.dimensionality)

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

        # collect the config from the attributes
        config = self.get_config()

        # change to list of channels (FIXME)
        config["channels"] = [ x.type.tensor_type.shape.dim[1].dim_value for x in self.inputs ]

        # add the additional parameters
        config["ports"] = len(self.inputs)

        # initialise layer
        return LayerBase.build("concat", config, self.backend, self.dimensionality)

    def get_edges_in(self, model):
        try:
            edges = []
            prev_nodes = filter(lambda x: x.output[0] in self.node.input, model.graph.node)
            for prev_node in prev_nodes:
                edges.append((onnx_helper.format_onnx_name(prev_node), self.name))
            return edges
        except StopIteration:
            return []

