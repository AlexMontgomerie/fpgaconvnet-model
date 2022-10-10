import copy
import importlib
import os
import random

from graphviz import Digraph
import networkx as nx
import onnx
from onnxsim import simplify
import onnx.numpy_helper
import onnx.utils
import onnxoptimizer as optimizer
import pydot
import numpy as np

import fpgaconvnet.tools.graphs as graphs

import fpgaconvnet.parser.onnx.helper as onnx_helper
import fpgaconvnet.parser.onnx.parse as onnx_parse
import fpgaconvnet.parser.onnx.passes as onnx_passes

from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type

from fpgaconvnet.parser.onnx.parse import *

class Parser:

    def __init__(self):

        # passes for onnx optimizer
        self.onnxoptimizer_passes = [
            "fuse_bn_into_conv",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
            "fuse_matmul_add_bias_into_gemm",
            "set_unique_name_for_nodes",
        ]

        # minimum supported opset version
        self.opset_version =  25

        # set the backend string
        self.backend = "hls"

        # quantisation mode [ float, QDQ, BFP, config ]
        self.quant_mode = "fake_float"

    def optimize_onnx(self, model, passes):
        model_opt = model
        for opt_pass in passes:
            model_opt = getattr(onnx_passes, opt_pass)(model_opt)
        return model_opt

    def load_onnx_model(self, onnx_filepath):

        # load onnx model
        model = onnx.load(onnx_filepath)

        # simplify model
        model, _ = simplify(model)

        # validate model
        onnx.checker.check_model(model)

        # # check opset version
        # assert model.opset_import.version >= self.opset_version, f"ONNX Operator version {model.opset_import.version} not supported!"

        # remove doc strings
        onnx.helper.strip_doc_string(model)

        # add inputs from initializers
        onnx_helper.add_input_from_initializer(model) #Seems to be necessary for conv layers from pytorch (at least)

        # perform onnx optimization passes
        model_opt = optimizer.optimize(model,
                passes=self.onnxoptimizer_passes)

        # infer shapes before manual optimisations
        self.model_opt = onnx.shape_inference.infer_shapes(model_opt)

        # perform fpgaconvnet-based optimization passes
        model_opt = self.optimize_onnx(model_opt,
                ["fuse_matmul_add_into_gemm", "convert_matmul_to_gemm",
                    "remove_redundant_pooling", "remove_training_nodes",
                    "convert_pool_to_global_pool", "convert_reshape_to_flatten",
                    "convert_transpose_flatten_gemm_to_flatten_gemm"])

        # infer shapes of optimised model
        self.model_opt = onnx.shape_inference.infer_shapes(model_opt)

        # check optimized model
        onnx.checker.check_model(model_opt)

        # check that models are equivalent
        onnx_helper.check_model_equivalence(model, model_opt)

        return model_opt

    def validate_hardware_onnx_model(self, onnx_model):
        pass

    def get_hardware_from_onnx_node(self, graph, node):

        # register converters
        converter = {
            LAYER_TYPE.Convolution: ParseOnnxConvNode,
            LAYER_TYPE.InnerProduct: ParseOnnxInnerProductNode,
            LAYER_TYPE.Pooling: ParseOnnxPoolingNode,
            LAYER_TYPE.ReLU: ParseOnnxReLUNode,
            LAYER_TYPE.NOP: ParseOnnxNOPNode,
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

            # get edges from the hardware
            for edge in hardware.get_edges_out(onnx_model):
                graph.add_edge(*edge)

        # return the graph
        return graph

    def prototxt_to_fpgaconvnet(self, proto_filepath):
        pass

    def rename_graph(self):
        pass

if __name__ == "__main__":

    p = Parser()

    # print("parsing alexnet")
    # p.onnx_to_fpgaconvnet("../samo/models/alexnet.onnx")

    print("Keras-converted models:")

    print(" - parsing mpcnn")
    p.onnx_to_fpgaconvnet("models/mpcnn.onnx")

    print(" - parsing cnv")
    p.onnx_to_fpgaconvnet("models/cnv.onnx")

    print(" - parsing simple")
    p.onnx_to_fpgaconvnet("models/simple.onnx")

    print(" - parsing lfc")
    p.onnx_to_fpgaconvnet("models/lfc.onnx")

#     print("parsing mobilenetv2")
#     p.onnx_to_fpgaconvnet("models/mobilenetv2-7.onnx")

    print("parsing mpcnn")
    p.onnx_to_fpgaconvnet("models/mpcnn.onnx")

    print("parsing vgg11")
    p.onnx_to_fpgaconvnet("models/vgg11.onnx")

    # # print("parsing vgg16")
    # # p.onnx_to_fpgaconvnet("models/vgg16-7.onnx")

    # # print("parsing zfnet")
    # # p.onnx_to_fpgaconvnet("models/zfnet512-3.onnx")

    # # print("parsing key word spotting network")
    # # p.onnx_to_fpgaconvnet("models/kws.onnx")

    # print("parsing mobilenetv1 shrunk")
    # p.onnx_to_fpgaconvnet("models/vww.onnx")

    print("parsing resnet 8")
    p.onnx_to_fpgaconvnet("models/resnet8.onnx")

