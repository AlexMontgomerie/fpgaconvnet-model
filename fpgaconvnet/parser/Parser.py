import copy
import sys
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
            # "fuse_bn_into_conv",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
            "fuse_matmul_add_bias_into_gemm",
            "set_unique_name_for_nodes",
        ]

        # minimum supported opset version
        self.onnx_opset_version = 12

        # set the backend string
        self.backend = "hls"

        # quantisation mode [ float, QDQ, BFP, config ]
        self.quant_mode = "fake_float"

        # batch size
        self.batch_size = 1

    def optimize_onnx(self, model, passes):
        model_opt = model
        for opt_pass in passes:
            model_opt = getattr(onnx_passes, opt_pass)(model_opt)
        return model_opt

    def load_onnx_model(self, onnx_filepath):

        # load onnx model
        model = onnx.load(onnx_filepath)

        # update model's batch size
        model = onnx_helper.update_batch_size(model, self.batch_size)

        # simplify model
        model_opt, _ = simplify(model)
        # model_opt, _ = (model, False)

        # validate model
        onnx.checker.check_model(model_opt)

        # # check opset version
        # assert model.opset_import.version >= self.opset_version, f"ONNX Operator version {model.opset_import.version} not supported!"

        # remove doc strings
        onnx.helper.strip_doc_string(model_opt)

        # add inputs from initializers
        onnx_helper.add_input_from_initializer(model_opt) #Seems to be necessary for conv layers from pytorch (at least)

        # perform onnx optimization passes
        model_opt = optimizer.optimize(model_opt,
                passes=self.onnxoptimizer_passes)

        # infer shapes before manual optimisations
        model_opt = onnx.shape_inference.infer_shapes(model_opt)

        # perform fpgaconvnet-based optimization passes
        model_opt = self.optimize_onnx(model_opt,
                ["convert_matmul_to_gemm", "remove_redundant_pooling",
                    "make_clip_min_max_scalar", "remove_training_nodes",
                    "convert_pool_to_global_pool", "convert_reshape_to_flatten",
                    "convert_transpose_flatten_gemm_to_flatten_gemm",
                    "rename_all_nodes"])

        # infer shapes of optimised model
        self.model_opt = onnx.shape_inference.infer_shapes(model_opt)

        # onnx.save(model_opt, "model_opt.onnx")

        # check optimized model
        onnx.checker.check_model(model_opt)

        # check that models are equivalent
        onnx_helper.check_model_equivalence(model, model_opt)

        return model_opt

    def get_hardware_from_onnx_node(self, graph, node):

        # register converters
        converter = {
            LAYER_TYPE.Convolution: ParseOnnxConvNode,
            LAYER_TYPE.InnerProduct: ParseOnnxInnerProductNode,
            LAYER_TYPE.Pooling: ParseOnnxPoolingNode,
            LAYER_TYPE.AveragePooling: ParseOnnxAveragePoolingNode,
            LAYER_TYPE.EltWise: ParseOnnxEltWiseNode,
            LAYER_TYPE.ReLU: ParseOnnxReLUNode,
            LAYER_TYPE.NOP: ParseOnnxNOPNode,
        }

        # get the node type
        node_type = from_onnx_op_type(node.op_type)
        # try converter
        try:
            return converter[node_type](graph, node, backend=self.backend)
        except KeyError:
            raise TypeError(f"{node_type} not supported, exiting now")

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
            graph.add_node(hardware.name,
                    **hardware.get_node_info())

            # get edges from the hardware
            for edge in hardware.get_edges_in(onnx_model):
                graph.add_edge(*edge)

        # return the graph
        return onnx_model, graph

    def prototxt_to_fpgaconvnet(self, proto_filepath):
        pass

    def rename_graph(self):
        pass

if __name__ == "__main__":

    p = Parser()

    # print("parsing alexnet")
    # p.onnx_to_fpgaconvnet("../samo/models/alexnet.onnx")

    # print("Keras-converted models:")
    # print(f" - parsing cnv")
    # p.onnx_to_fpgaconvnet(f"models/from_keras/cnv.onnx")
    # print(f" - parsing mpcnn")
    # p.onnx_to_fpgaconvnet(f"models/from_keras/mpcnn.onnx")
    # print(f" - parsing sfc")
    # p.onnx_to_fpgaconvnet(f"models/from_keras/sfc.onnx")
    # print(f" - parsing vgg11")
    # p.onnx_to_fpgaconvnet(f"models/from_keras/vgg11.onnx")
    print(f" - parsing resnet18")
    model, graph = p.onnx_to_fpgaconvnet(f"models/from_keras/resnet18.onnx")

    # for model in os.listdir("models/from_keras/"):
    #     print(f" - parsing {model}")
    #     p.onnx_to_fpgaconvnet(f"models/from_keras/{model}")

    # print("Pytorch-converted models:")
    # print(f" - parsing alexnet_cifar")
    # p.onnx_to_fpgaconvnet(f"models/from_pytorch/alexnet_cifar.onnx")
    # print(f" - parsing vgg16_cifar")
    # p.onnx_to_fpgaconvnet(f"models/from_pytorch/vgg16_cifar.onnx")

    # print("Pytorch-converted models:")
    # for model in os.listdir("models/from_pytorch/"):
    #     print(f" - parsing {model}")
    #     p.onnx_to_fpgaconvnet(f"models/from_pytorch/{model}")

    print("ONNX model zoo models:")
    # print(f" - parsing vgg16")
    # p.onnx_to_fpgaconvnet(f"models/from_onnx_model_zoo/vgg16-12.onnx")
    # print(f" - parsing mobilenetv2")
    # p.onnx_to_fpgaconvnet(f"models/from_onnx_model_zoo/mobilenetv2-12.onnx")
    # print(f" - parsing resnet18")
    # p.onnx_to_fpgaconvnet(f"models/from_onnx_model_zoo/resnet18-12.onnx")

    # print("ONNX model zoo models:")
    # for model in os.listdir("models/from_onnx_model_zoo/"):
    #     print(f" - parsing {model}")
    #     p.onnx_to_fpgaconvnet(f"models/from_onnx_model_zoo/{model}")

    # print("3D models:")
    # print(f" - parsing x3d_m")
    # p.onnx_to_fpgaconvnet(f"models/3d/x3d_m.onnx")
    # print(f" - parsing mobilenetv2")
    # p.onnx_to_fpgaconvnet(f"models/from_onnx_model_zoo/mobilenetv2-12.onnx")

    # print("ONNX model zoo models:")
    # for model in os.listdir("models/from_onnx_model_zoo/"):
    #     print(f" - parsing {model}")
    #     p.onnx_to_fpgaconvnet(f"models/from_onnx_model_zoo/{model}")


