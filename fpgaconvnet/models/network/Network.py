import os
import json
import pydot
import copy
import math
import numpy as np
import networkx as nx

from google.protobuf import json_format
import fpgaconvnet.proto.fpgaconvnet_pb2

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.matrix as matrix
import fpgaconvnet.tools.helper as helper

import fpgaconvnet.tools.layer_enum
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.models.layers import ConvolutionLayer
from fpgaconvnet.models.layers import InnerProductLayer
from fpgaconvnet.models.layers import PoolingLayer
from fpgaconvnet.models.layers import ReLULayer
from fpgaconvnet.models.layers import SqueezeLayer
from fpgaconvnet.models.partition.Partition import Partition

from fpgaconvnet.parser import Parser

class Network():

    def __init__(self, name, network_path, batch_size=1, freq=125,
            reconf_time=0.0, data_width=16, weight_width=8, acc_width=30,
            fuse_bn=True, rsc_allocation=1.0):

        # empty transforms configuration
        self.transforms_config = {}

        ## percentage resource allocation
        self.rsc_allocation = rsc_allocation

        ## bitwidths
        self.data_width     = data_width
        self.weight_width   = weight_width
        self.acc_width      = acc_width

        # network name
        self.name = name

        # initialise variables
        self.batch_size = batch_size
        self.fuse_bn = fuse_bn

        # load network
        self.parser = Parser()
        self.model, self.graph = self.parser.onnx_to_fpgaconvnet(network_path)
        # self.model, self.graph = parser.parse_net(network_path, view=False,
        #         data_width=self.data_width, weight_width=self.weight_width,
        #         acc_width=self.acc_width, fuse_bn=self.fuse_bn)

        # node and edge lists
        self.node_list = list(self.graph.nodes())
        self.edge_list = list(self.graph.edges())

        # matrices
        self.connections_matrix = matrix.get_connections_matrix(self.graph)
        self.workload_matrix    = matrix.get_workload_matrix(self.graph)

        # partitions
        self.partitions = [ Partition(copy.deepcopy(self.graph),
                data_width=self.data_width, weight_width=self.weight_width,
                acc_width=self.acc_width) ]

        # platform
        self.platform = {
            'name'          : 'platform',
            'freq'          : freq,
            'reconf_time'   : 0.0,
            'wr_time'       : 0.0,
            'ports'         : 4,
            'port_width'    : 64,
            'mem_bandwidth' : 0,
            'mem_capacity'  : 0,
            'constraints'   : {
                'FF'    : 0,
                'LUT'   : 0,
                'DSP'   : 0,
                'BRAM'  : 0
            }
        }

        # all types of layers
        self.conv_layers = helper.get_all_layers(self.graph, LAYER_TYPE.Convolution)
        self.pool_layers = helper.get_all_layers(self.graph, LAYER_TYPE.Pooling)

        # update partitions
        self.update_partitions()


    # from fpgaconvnet.transforms.partition import check_parallel_block
    # from fpgaconvnet.transforms.partition import get_all_horizontal_splits
    # from fpgaconvnet.transforms.partition import get_all_vertical_splits
    # from fpgaconvnet.transforms.partition import get_all_horizontal_merges
    # from fpgaconvnet.transforms.partition import get_all_vertical_merges
    # from fpgaconvnet.transforms.partition import split_horizontal
    # from fpgaconvnet.transforms.partition import split_vertical
    # from fpgaconvnet.transforms.partition import merge_horizontal
    # from fpgaconvnet.transforms.partition import merge_vertical
    # from fpgaconvnet.transforms.partition import split_horizontal_complete
    # from fpgaconvnet.transforms.partition import split_vertical_complete
    # from fpgaconvnet.transforms.partition import split_complete
    # from fpgaconvnet.transforms.partition import merge_horizontal_complete
    # from fpgaconvnet.transforms.partition import merge_vertical_complete
    # from fpgaconvnet.transforms.partition import merge_complete
    # from fpgaconvnet.transforms.partition import apply_random_partition

    from fpgaconvnet.models.network.report import create_report

    from fpgaconvnet.models.network.scheduler import get_partition_order
    from fpgaconvnet.models.network.scheduler import get_input_base_addr
    from fpgaconvnet.models.network.scheduler import get_output_base_addr
    from fpgaconvnet.models.network.scheduler import get_partition_input_dependence
    from fpgaconvnet.models.network.scheduler import get_partition_output_dependence
    from fpgaconvnet.models.network.scheduler import get_scheduler
    from fpgaconvnet.models.network.scheduler import get_schedule_csv
    from fpgaconvnet.models.network.scheduler import check_scheduler

    from fpgaconvnet.models.network.update import update_partitions
    from fpgaconvnet.models.network.update import update_platform
    from fpgaconvnet.models.network.update import update_coarse_in_out_partition

    from fpgaconvnet.models.network.represent import get_model_input_node
    from fpgaconvnet.models.network.represent import get_model_output_node
    from fpgaconvnet.models.network.represent import get_stream_in_coarse
    from fpgaconvnet.models.network.represent import get_stream_out_coarse
    from fpgaconvnet.models.network.represent import save_all_partitions

    from fpgaconvnet.models.network.validate import check_ports
    from fpgaconvnet.models.network.validate import check_resources
    from fpgaconvnet.models.network.validate import get_resources_bad_partitions
    from fpgaconvnet.models.network.validate import check_workload
    from fpgaconvnet.models.network.validate import check_streams
    from fpgaconvnet.models.network.validate import check_partitions
    from fpgaconvnet.models.network.validate import check_memory_bandwidth

    def get_memory_usage_estimate(self):

        # for sequential networks, our worst-case memory usage is
        # going to be both the largest input and output featuremap pair

        # get the largest input featuremap size
        max_input_size = 0
        max_output_size = 0
        for partition in self.partitions:
            input_node  = partition.input_nodes[0]
            output_node = partition.output_nodes[0]
            partition_input_size  = partition.graph.nodes[input_node]['hw'].workload_in()*partition.batch_size
            partition_output_size = partition.graph.nodes[output_node]['hw'].workload_out()*partition.batch_size*partition.wr_factor
            if partition_input_size > max_input_size:
                max_input_size = partition_input_size
            if partition_output_size > max_output_size:
                max_output_size = partition_output_size

        return math.ceil(((max_input_size + max_output_size)*self.data_width)/8)

    def get_latency(self, partition_list=None):
        if partition_list == None:
            partition_list = list(range(len(self.partitions)))
        latency = 0
        # iterate over partitions:
        for partition_index, partition in enumerate(self.partitions):
            if partition_index not in partition_list:
                continue
            # accumulate latency for each partition
            latency += partition.get_latency(self.platform["freq"])
        # return the total latency as well as reconfiguration time
        return latency + (len(partition_list)-1)*self.platform["reconf_time"]

    def get_throughput(self, partition_list=None):
        if partition_list == None:
            partition_list = list(range(len(self.partitions)))
        # return the frames per second
        return float(self.batch_size)/self.get_latency(partition_list)

    def visualise(self, output_path, mode="dot"):
        g = pydot.Dot(graph_type='digraph', splines="ortho", fontsize=25)
        main_cluster = pydot.Cluster("network", label="Network")

        cluster_nodes = []

        # get all partitions
        for partition in self.partitions:
            # add subgraph
            partition_cluster, input_node, output_node = partition.visualise(self.partitions.index(partition))
            main_cluster.add_subgraph(partition_cluster)

        # connect each partition with a reconfiguration node
        for i in range(len(self.partitions)-1):
            # add reconfiguration node
            reconf_name = f"reconf_{i}"
            main_cluster.add_node(pydot.Node(reconf_name, label="RECONFIGURATION",
                shape="plaintext", fontsize=50))
            # add edges between reconfiguration nodes
            main_cluster.add_edge(pydot.Edge(f"mem_write_{i}", reconf_name))
            main_cluster.add_edge(pydot.Edge(reconf_name, f"mem_read_{i+1}"))

        # add main cluster
        g.add_subgraph(main_cluster)

        # save graph
        if mode == "dot":
            g.write_raw(output_path)
        elif mode == "svg":
            g.write_svg(output_path)
        elif mode == "png":
            g.write_png(output_path)
        else:
            raise TypeError

    def get_layer_hardware(self, layer_proto):
        # get layer type
        layer_type = fpgaconvnet.tools.layer_enum.from_proto_layer_type(layer_proto.type)
        # Convolution layer
        if layer_type == LAYER_TYPE.Convolution:
            return ConvolutionLayer(
                layer_proto.parameters.channels_out,
                layer_proto.parameters.rows_in,
                layer_proto.parameters.cols_in,
                layer_proto.parameters.channels_in,
                kernel_size =list(layer_proto.parameters.kernel_size),
                stride      =list(layer_proto.parameters.stride),
                pad         = [
                    layer_proto.parameters.pad_top,
                    layer_proto.parameters.pad_right,
                    layer_proto.parameters.pad_bottom,
                    layer_proto.parameters.pad_left],
                groups      =layer_proto.parameters.groups,
                fine        =layer_proto.parameters.fine,
                coarse_in   =layer_proto.parameters.coarse_in,
                coarse_out  =layer_proto.parameters.coarse_out
            )

        # Inner Product Layer
        if layer_type == LAYER_TYPE.InnerProduct:
            return InnerProductLayer(
                layer_proto.parameters.channels_out,
                layer_proto.parameters.rows_in,
                layer_proto.parameters.cols_in,
                layer_proto.parameters.channels_in,
                coarse_in   =layer_proto.parameters.coarse_in,
                coarse_out  =layer_proto.parameters.coarse_out
            )

        # Pooling layer
        if layer_type == LAYER_TYPE.Pooling:
            return PoolingLayer(
                layer_proto.parameters.rows_in,
                layer_proto.parameters.cols_in,
                layer_proto.parameters.channels_in,
                pool_type   = 'max',
                kernel_size =list(layer_proto.parameters.kernel_size),
                stride      =list(layer_proto.parameters.stride),
                pad         = [
                    layer_proto.parameters.pad_top,
                    layer_proto.parameters.pad_right,
                    layer_proto.parameters.pad_bottom,
                    layer_proto.parameters.pad_left],
                coarse      =layer_proto.parameters.coarse
            )

        # ReLU Layer
        if layer_type == LAYER_TYPE.ReLU:
            # create relu layer hardware
            return ReLULayer(
                layer_proto.parameters.rows_in,
                layer_proto.parameters.cols_in,
                layer_proto.parameters.channels_in,
                coarse      =layer_proto.parameters.coarse
            )

        # Squeeze Layer
        if layer_type == LAYER_TYPE.Squeeze:
            # create relu layer hardware
            return SqueezeLayer(
                layer_proto.parameters.rows_in,
                layer_proto.parameters.cols_in,
                layer_proto.parameters.channels_in,
                coarse_in   =layer_proto.parameters.coarse_in,
                coarse_out  =layer_proto.parameters.coarse_out
            )

    def load_network(self, network_path):
        # load the prototxt file
        partitions = fpgaconvnet.proto.fpgaconvnet_pb2.partitions()
        with open(network_path, "r") as f:
            json_format.Parse(f.read(), partitions)
        # delete current partitions
        self.partitions = []
        # iterate over partitions
        for i, partition in enumerate(partitions.partition):
            # add all layers to partition
            graph = nx.DiGraph()
            for layer in partition.layers:
                # get layer type and hardware
                layer_type = fpgaconvnet.tools.layer_enum.from_proto_layer_type(layer.type)
                layer_hw = self.get_layer_hardware(layer)
                # add layer
                graph.add_node( layer.name, type=layer_type, hw=layer_hw, inputs={} )
            # add all connections to graph
            for layer in partition.layers:
                if layer.node_in != layer.name:
                    graph.add_edge(layer.node_in, layer.name)
                if layer.node_out != layer.name:
                    graph.add_edge(layer.name, layer.node_out)
            # add partition
            new_partition = Partition(graph)
            # update partition attributes
            new_partition.wr_factor = int(partition.weights_reloading_factor)
            new_partition.wr_layer  = partition.weights_reloading_layer
            self.partitions.append(new_partition)

