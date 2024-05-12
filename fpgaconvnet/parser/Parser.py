import importlib

import networkx as nx
import onnx
import onnx.numpy_helper
import onnx.utils
import onnxoptimizer as optimizer
import pydot
from google.protobuf import json_format
from onnxsim import simplify

import fpgaconvnet.parser.onnx.helper as onnx_helper
import fpgaconvnet.parser.onnx.passes as onnx_passes
import fpgaconvnet.proto.fpgaconvnet_pb2
import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.models.layers import SplitLayer, SplitLayer3D
from fpgaconvnet.models.network import Network
from fpgaconvnet.models.partition import Partition
from fpgaconvnet.parser.onnx.parse import *
from fpgaconvnet.parser.prototxt.parse import *
from fpgaconvnet.parser.quant.int import get_scale_shift_node
from fpgaconvnet.tools.layer_enum import (LAYER_TYPE, from_onnx_op_type,
                                          from_proto_layer_type)


class Parser:

    def __init__(self, backend="chisel", regression_model="linear_regression",
            quant_mode="auto", batch_size=1, convert_gemm_to_conv=False, custom_onnx=False):

        # set the backend string
        self.backend = backend

        # set the regression model
        self.regression_model = regression_model

        # quantisation mode [ auto, float, QDQ, BFP, config ]
        self.quant_mode = quant_mode

        # batch size
        self.batch_size = batch_size

        # custom onnx flag
        self.custom_onnx = custom_onnx

        # passes for onnx optimizer
        self.onnxoptimizer_passes = [
            "extract_constant_to_initializer",
            "fuse_bn_into_conv",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
            "fuse_matmul_add_bias_into_gemm",
            "set_unique_name_for_nodes",
            "eliminate_nop_pad",
            "fuse_pad_into_conv",
            "fuse_pad_into_pool",
        ]

        # passes for fpgaconvnet onnx optimizer
        self.fpgaconvnet_pre_onnx_passes = [
            # "absorb_quantise",
            "convert_to_version_15",
            "fuse_mul_add_into_bn",
        ]

        self.fpgaconvnet_post_onnx_passes = [
            "eliminate_nop_pad",
            "fuse_mul_sigmoid_into_hardswish",
            "fuse_add_clip_mul_div_into_hardswish",
            "fuse_matmul_add_into_gemm",
            "convert_matmul_to_gemm",
            "fuse_bn_into_gemm",
            "eliminate_nop_pool",
            "make_clip_min_max_scalar",
            "remove_training_nodes",
            "convert_pool_to_global_pool",
            "convert_reshape_to_flatten",
            "convert_transpose_flatten_gemm_to_flatten_gemm",
            "rename_all_nodes",
            "move_relu_after_quant",
            "add_nop_to_split_output",
            # "remove_empty_inputs_outputs",
        ]

        self.fpgaconvnet_post_quant_passes = [
            # "insert_scale_shift_quant",
            "remove_quant_nodes",
        ]

        # minimum supported opset version
        self.onnx_opset_version = 14

        # flag to convert InnerProduct to Convolution
        self.convert_gemm_to_conv = convert_gemm_to_conv

    def add_onnx_optimization_passes(self, passes):
        for pass_name in passes:
            self.fpgaconvnet_pre_onnx_passes.append(pass_name)

    def optimize_onnx(self, model, passes):
        model_opt = model
        for opt_pass in passes:
            model_opt = getattr(onnx_passes, opt_pass)(model_opt)
        return model_opt

    def load_onnx_model(self, onnx_filepath):

        # load onnx model
        model = onnx.load(onnx_filepath)
        input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] # We assume the model has only one input
        dimensionality = len(input_shape) - 2

        # update model's batch size
        model = onnx_helper.update_batch_size(model, self.batch_size)

        if self.custom_onnx:
            model_opt = model
        else:
            # simplify model
            model_opt, _ = simplify(model)

            # validate model
            onnx.checker.check_model(model_opt)

        # remove doc strings
        onnx.helper.strip_doc_string(model_opt)

        # add inputs from initializers
        onnx_helper.add_input_from_initializer(model_opt) #Seems to be necessary for conv layers from pytorch (at least)

        # perform fpgaconvnet-based optimization passes (pre onnx optimizations)
        model_opt = self.optimize_onnx(model_opt, self.fpgaconvnet_pre_onnx_passes)

        # perform onnx optimization passes
        model_opt = optimizer.optimize(model_opt,
                passes=self.onnxoptimizer_passes)

        # perform fpgaconvnet-based optimization passes (post onnx optimizations)
        model_opt = self.optimize_onnx(model_opt, self.fpgaconvnet_post_onnx_passes)

        # infer shapes of optimised model
        model_opt = onnx.shape_inference.infer_shapes(model_opt)

        if not self.custom_onnx:
            # check optimized model
            onnx.checker.check_model(model_opt)

        return model_opt, dimensionality

    def get_hardware_from_onnx_node(self, graph, node, quant_format, dimensionality):

        # register converters
        converter = {
            LAYER_TYPE.Convolution: ParseOnnxConvNode,
            LAYER_TYPE.InnerProduct: ParseOnnxInnerProductNode,
            LAYER_TYPE.Pooling: ParseOnnxPoolingNode,
            LAYER_TYPE.GlobalPooling: ParseOnnxGlobalPoolingNode,
            LAYER_TYPE.EltWise: ParseOnnxEltWiseNode,
            LAYER_TYPE.Concat: ParseOnnxConcatNode,
            LAYER_TYPE.ReLU: ParseOnnxReLUNode,
            LAYER_TYPE.ThresholdedReLU: ParseOnnxThresholdedReLUNode,
            LAYER_TYPE.Sigmoid: ParseOnnxActivationNode,
            LAYER_TYPE.ReSize: ParseOnnxReSizeNode,
            LAYER_TYPE.HardSigmoid: ParseOnnxActivationNode,
            LAYER_TYPE.HardSwish: ParseOnnxHardSwishNode,
            LAYER_TYPE.Chop: ParseOnnxChopNode,
            LAYER_TYPE.Reshape: ParseOnnxNOPNode,
            LAYER_TYPE.Pad: ParseOnnxNOPNode,
            LAYER_TYPE.NOP: ParseOnnxNOPNode,
        }

        # get the node type
        node_type = from_onnx_op_type(node.op_type)

        # try converter
        try:
            return converter[node_type](graph, node, quant_format, dimensionality,
                    backend=self.backend, regression_model=self.regression_model,
                    convert_gemm_to_conv=self.convert_gemm_to_conv)
        except KeyError:
            raise TypeError(f"{node_type} not supported, exiting now")

    def get_quantisation(self, model, **kwargs):

        # get the quantisation method
        try:
            quant = importlib.import_module(f"fpgaconvnet.parser.quant.{self.quant_mode}")
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"quantisation mode {self.quant_mode} not supported")

        # get the quantisation format
        quant_format = quant.get_quant_param(model, **kwargs)

        # perform fpgaconvnet-based optimization passes (post quantisation)
        model_opt = self.optimize_onnx(model, self.fpgaconvnet_post_quant_passes)

        # return model and quantisation
        return model_opt, quant_format

    def add_split(self, graph, dimensionality):
        # iterate over nodes in the graph
        nodes = list(graph.nodes())
        for node in nodes:
            # get the nodes out
            nodes_out = graphs.get_next_nodes(graph, node)
            # add a split layer if there are more than 1 nodes out
            if len(nodes_out) > 1 and not graph.nodes[node]['type'] == LAYER_TYPE.Chop:
                # create a split node
                split_node  = f"{node}_split"
                if dimensionality == 2:
                    graph.add_node(split_node,
                        type=LAYER_TYPE.Split,
                        onnx_node=graph.nodes[node]["onnx_node"],
                        onnx_input=graph.nodes[node]["onnx_input"],
                        onnx_output=graph.nodes[node]["onnx_output"],
                        hw=SplitLayer(
                            graph.nodes[node]['hw'].rows_out(),
                            graph.nodes[node]['hw'].cols_out(),
                            graph.nodes[node]['hw'].channels_out(),
                            graph.nodes[node]['hw'].streams_out(),
                            len(nodes_out),
                            data_t=graph.nodes[node]['hw'].data_t,
                            input_compression_ratio=graph.nodes[node]['hw'].output_compression_ratio,
                            output_compression_ratio=[graph.nodes[node]['hw'].output_compression_ratio[0]]*len(nodes_out),
                        )
                    )
                elif dimensionality == 3:
                    graph.add_node(split_node,
                        type=LAYER_TYPE.Split,
                        onnx_node=graph.nodes[node]["onnx_node"],
                        onnx_input=graph.nodes[node]["onnx_input"],
                        onnx_output=graph.nodes[node]["onnx_output"],
                        hw=SplitLayer3D(
                            graph.nodes[node]['hw'].rows_out(),
                            graph.nodes[node]['hw'].cols_out(),
                            graph.nodes[node]['hw'].depth_out(),
                            graph.nodes[node]['hw'].channels_out(),
                            graph.nodes[node]['hw'].streams_out(),
                            len(nodes_out),
                            data_t=graph.nodes[node]['hw'].data_t,
                            input_compression_ratio=graph.nodes[node]['hw'].output_compression_ratio,
                            output_compression_ratio=[graph.nodes[node]['hw'].output_compression_ratio[0]]*len(nodes_out),
                        )
                    )
                # iterate over nodes out
                for node_out in nodes_out:
                    # remove edge from original node
                    graph.remove_edge(node, node_out)
                    # add edge to split node
                    graph.add_edge(split_node, node_out)
                # add edge from original node to split node
                graph.add_edge(node, split_node)
        return graph

    def onnx_to_fpgaconvnet(self, onnx_filepath, save_opt_model=True):
        # load the onnx model
        onnx_model, dimensionality = self.load_onnx_model(onnx_filepath)

        # get the quantisation parameters
        onnx_model, quant_format = self.get_quantisation(onnx_model)

        if save_opt_model:
            optimize_onnx_filepath = f"{onnx_filepath.split('.onnx')[0]}_optimized.onnx"
            onnx.save(onnx_model, optimize_onnx_filepath)

        # create a networkx graph
        graph = nx.DiGraph()

        # extra quantisation nodes
        extra_quant_nodes = []

        # add nodes from onnx to the graph
        for node in onnx_model.graph.node:
            # get the node name
            node_name = onnx_helper.format_onnx_name(node)

            # get the hardware for the node
            hardware = self.get_hardware_from_onnx_node(
                    onnx_model.graph, node, quant_format[node_name], dimensionality)

            # add node to graph
            graph.add_node(hardware.name, **hardware.get_node_info())

            # get edges from the hardware
            for edge in hardware.get_edges_in(onnx_model):
                graph.add_edge(*edge)

            # add extra quantisation hardware
            # if "weight_quant" in quant_format[node_name]: # TODO: check all in
            if self.quant_mode == "int" and hardware.layer_type in [LAYER_TYPE.Convolution,
                    LAYER_TYPE.InnerProduct, LAYER_TYPE.GlobalPooling ]: # TODO: check all in
                extra_quant_nodes.append((hardware, quant_format[node_name]))

        # add the extra quantisation nodes
        for node, quant_format in extra_quant_nodes:

            # get the batch norm node
            bn_node = get_scale_shift_node(quant_format, node)

            # add node to graph
            graph.add_node(f"{node.name}_scale_shift", **bn_node)

            # insert the bn node after the node
            for node_out in graphs.get_next_nodes(graph, node.name):

                # remove previous edge, and add new
                graph.add_edge(f"{node.name}_scale_shift", node_out)
                graph.remove_edge(node.name, node_out)

            # connect batch norm to node
            graph.add_edge(node.name, f"{node.name}_scale_shift")

        # add split nodes to the graph
        graph = self.add_split(graph, dimensionality)

        # remove NOP nodes from the graph
        graph = self.remove_node_by_type(graph, LAYER_TYPE.NOP)
        # graph = self.remove_node_by_type(graph, LAYER_TYPE.Reshape)

        # return the graph
        return Network("from_onnx", onnx_model, graph, dimensionality=dimensionality)

    def get_hardware_from_prototxt_node(self, node, dimensionality):

        # register converters
        converter = {
            LAYER_TYPE.Convolution: ParsePrototxtConvNode,
            LAYER_TYPE.InnerProduct: ParsePrototxtInnerProductNode,
            LAYER_TYPE.Pooling: ParsePrototxtPoolingNode,
            LAYER_TYPE.GlobalPooling: ParsePrototxtGlobalPoolingNode,
            LAYER_TYPE.EltWise: ParsePrototxtEltWiseNode,
            LAYER_TYPE.ReLU: ParsePrototxtReLUNode,
            LAYER_TYPE.ThresholdedReLU: ParsePrototxtThresholdedReLUNode,
            LAYER_TYPE.Squeeze: ParsePrototxtSqueezeNode,
            LAYER_TYPE.Split: ParsePrototxtSplitNode,
            LAYER_TYPE.Concat: ParsePrototxtConcatNode,
            LAYER_TYPE.Sigmoid: ParsePrototxtActivationNode,
            LAYER_TYPE.ReSize: ParsePrototxtReSizeNode,
            LAYER_TYPE.HardSigmoid: ParsePrototxtActivationNode,
            LAYER_TYPE.HardSwish: ParsePrototxtHardSwishNode,
            LAYER_TYPE.Chop: ParsePrototxtChopNode,
        }

        # get the node type
        node_type = from_proto_layer_type(node.type)


        # try converter
        try:
            return converter[node_type](node, dimensionality, backend=self.backend,
                    regression_model=self.regression_model)
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
            if i == 0:
                net.batch_size = partition.batch_size
            # add all layers to partition
            graph = nx.DiGraph()
            for layer in partition.layers:

                # get the hardware for the node
                hardware = self.get_hardware_from_prototxt_node(layer, net.dimensionality)

                # todo: move this inside get_hardware_from_prototxt_node
                hardware.hw.stream_inputs = hardware.attr["stream_inputs"]
                hardware.hw.stream_outputs = hardware.attr["stream_outputs"]

                # add node to graph
                graph.add_node( layer.name, **hardware.get_node_info(net.graph) )

                # get edges from the hardware
                for edge in hardware.get_edges_in():
                    graph.add_edge(*edge)

            # add partition
            new_partition = Partition(graph, net.dimensionality, batch_size=partition.batch_size)

            # update partition attributes
            new_partition.wr_factor = int(partition.weights_reloading_factor)
            new_partition.wr_layer  = partition.weights_reloading_layer
            net.partitions.append(new_partition)

        # return updated network
        return net

    def remove_node_by_type(self, graph, layer_type):
        # get input and output graphs
        input_nodes  = graphs.get_input_nodes(graph, allow_multiport=True)
        output_nodes = graphs.get_output_nodes(graph, allow_multiport=True)
        # remove input squeeze module
        for input_node in input_nodes:
            if input_node in graph.nodes:
                if graph.nodes[input_node]['type'] == layer_type:
                    graph.remove_node(input_node)
        # remove output squeeze module
        for output_node in output_nodes:
            if output_node in graph.nodes:
                if graph.nodes[output_node]['type'] == layer_type:
                    graph.remove_node(output_node)
        # remove intermediate squeeze modules
        remove_nodes = []
        for node in graph.nodes():
            if graph.nodes[node]['type'] == layer_type:
                # add squeeze nodes to list
                remove_nodes.append(node)
                # place edge back
                for prev_node in graphs.get_prev_nodes(graph, node):
                    for next_node in graphs.get_next_nodes(graph, node):
                        graph.add_edge(prev_node,next_node)
        # remove squeeze nodes
        graph.remove_nodes_from(remove_nodes)
        # return the graph
        return graph

if __name__ == "__main__":

    p = Parser()

    models = ["c3d", "r2plus1d", "slowonly", "x3d_m"]
    model_name = models[1]

    print(f" - parsing {model_name}")
    net = p.onnx_to_fpgaconvnet(f"tests/models/{model_name}.onnx")

    # print performance and resource estimates
    print(f"predicted latency (us): {net.get_latency()*1000000}")
    print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
    print(f"predicted resource usage: {net.partitions[0].get_resource_usage()}")

    # visualise the network configuration
    # net.visualise("image-path.png", mode="png")

    # export out the configuration
    # net.save_all_partitions("config-path.json")

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
    # print(f" - parsing resnet18")
    # net = p.onnx_to_fpgaconvnet(f"models/from_keras/lenet.onnx")

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

    # print("ONNX model zoo models:")
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


