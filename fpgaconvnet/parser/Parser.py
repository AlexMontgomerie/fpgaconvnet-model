from __future__ import absolute_import

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

# from fpgaconvnet.models.partition import Partition
# from fpgaconvnet.models.network import Network
from ..models.partition.Partition import Partition
from ..models.network.Network import Network


import fpgaconvnet.tools.graphs as graphs

import fpgaconvnet.parser.onnx.helper as onnx_helper
import fpgaconvnet.parser.onnx.parse as onnx_parse
import fpgaconvnet.parser.onnx.passes as onnx_passes

from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type, from_proto_layer_type

from fpgaconvnet.parser.onnx.parse import *
from fpgaconvnet.parser.prototxt.parse import *

from fpgaconvnet.parser.quant import quantise

from google.protobuf import json_format
import fpgaconvnet.proto.fpgaconvnet_pb2

class Parser:

    def __init__(self, backend="chisel", quant_mode="auto", batch_size=1):

        # set the backend string
        self.backend = backend

        # quantisation mode [ auto, float, QDQ, BFP, config ]
        self.quant_mode = quant_mode

        # batch size
        self.batch_size = batch_size

        # passes for onnx optimizer
        self.onnxoptimizer_passes = [
            "fuse_bn_into_conv",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
            "fuse_matmul_add_bias_into_gemm",
            "set_unique_name_for_nodes",
        ]

        # passes for fpgaconvnet onnx optimizer
        self.fpgaconvnet_pre_onnx_passes = [
            "absorb_quantise",
            "fuse_mul_add_into_bn",
        ]

        self.fpgaconvnet_post_onnx_passes = [
            "fuse_matmul_add_into_gemm",
            "convert_matmul_to_gemm",
            "fuse_bn_into_gemm",
            "remove_redundant_pooling",
            "make_clip_min_max_scalar",
            "remove_training_nodes",
            "convert_pool_to_global_pool",
            "convert_reshape_to_flatten",
            "convert_transpose_flatten_gemm_to_flatten_gemm",
            "rename_all_nodes"
        ]

        self.fpgaconvnet_post_quant_passes = [
            "remove_quant_nodes",
        ]

        # minimum supported opset version
        self.onnx_opset_version = 12

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

        # validate model
        onnx.checker.check_model(model_opt)

        # # check opset version
        # assert model.opset_import.version >= self.opset_version, f"ONNX Operator version {model.opset_import.version} not supported!"

        # remove doc strings
        onnx.helper.strip_doc_string(model_opt)

        # add inputs from initializers
        onnx_helper.add_input_from_initializer(model_opt) #Seems to be necessary for conv layers from pytorch (at least)

        # perform fpgaconvnet-based optimization passes (pre onnx optimizations)
        model_opt = self.optimize_onnx(model_opt, self.fpgaconvnet_pre_onnx_passes)

        # perform onnx optimization passes
        model_opt = optimizer.optimize(model_opt,
                passes=self.onnxoptimizer_passes)

        # infer shapes before manual optimisations
        model_opt = onnx.shape_inference.infer_shapes(model_opt)

        # perform fpgaconvnet-based optimization passes (post onnx optimizations)
        model_opt = self.optimize_onnx(model_opt, self.fpgaconvnet_post_onnx_passes)

        # infer shapes of optimised model
        self.model_opt = onnx.shape_inference.infer_shapes(model_opt)

        # check optimized model
        onnx.checker.check_model(model_opt)

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
            raise TypeError(f"{node.op_type} not supported, exiting now")

    def get_quantisation(self, model, **kwargs):

        # get the quantisation method
        quant = importlib.import_module(f"fpgaconvnet.parser.quant.{self.quant_mode}")

        # get the quantisation format
        quant_format = quant.get_quant_param(model)

        # perform fpgaconvnet-based optimization passes (post quantisation)
        model_opt = self.optimize_onnx(model, self.fpgaconvnet_post_quant_passes)

        # return model and quantisation
        return model_opt, quant_format

    def onnx_to_fpgaconvnet(self, onnx_filepath):

        # load the onnx model
        onnx_model = self.load_onnx_model(onnx_filepath)

        # get the quantisation parameters
        onnx_model, quant_format = self.get_quantisation(onnx_model)
        onnx.save(onnx_model, "model_opt.onnx")

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

        # apply quantisation to the graph
        quantise(graph, quant_format)

        # return the graph
        return Network("from_onnx", onnx_model, graph)

    def get_hardware_from_prototxt_node(self, node):

        # register converters
        converter = {
            LAYER_TYPE.Convolution: ParsePrototxtConvNode,
            LAYER_TYPE.InnerProduct: ParsePrototxtInnerProductNode,
            LAYER_TYPE.Pooling: ParsePrototxtPoolingNode,
            LAYER_TYPE.AveragePooling: ParsePrototxtAveragePoolingNode,
            LAYER_TYPE.EltWise: ParsePrototxtEltWiseNode,
            LAYER_TYPE.ReLU: ParsePrototxtReLUNode,
            LAYER_TYPE.Squeeze: ParsePrototxtSqueezeNode,
            LAYER_TYPE.Split: ParsePrototxtSplitNode,
        }

        # get the node type
        node_type = from_proto_layer_type(node.type)

        # try converter
        try:
            return converter[node_type](node, backend=self.backend)
        except KeyError:
            raise TypeError(f"{node_type} not supported, exiting now")


    def prototxt_to_fpgaconvnet(self, net, proto_filepath):

        # load the prototxt file
        partitions = fpgaconvnet.proto.fpgaconvnet_pb2.partitions()
        with open(proto_filepath, "r") as f:
            json_format.Parse(f.read(), partitions)

        # delete current partitions
        net.partitions = []

        # iterate over partitions
        for i, partition in enumerate(partitions.partition):

            # add all layers to partition
            graph = nx.DiGraph()
            for layer in partition.layers:

                # get the hardware for the node
                hardware = self.get_hardware_from_prototxt_node(layer)

                # add node to graph
                graph.add_node( layer.name, **hardware.get_node_info() )

                # get edges from the hardware
                for edge in hardware.get_edges_in():
                    graph.add_edge(*edge)

            # add partition
            new_partition = Partition(graph)

            # update partition attributes
            new_partition.wr_factor = int(partition.weights_reloading_factor)
            new_partition.wr_layer  = partition.weights_reloading_layer
            net.partitions.append(new_partition)

        # return updated network
        return net

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
    net = p.onnx_to_fpgaconvnet(f"models/from_keras/lenet.onnx")

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


