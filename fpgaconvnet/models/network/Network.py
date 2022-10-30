import os
import json
import pydot
import copy
import math
import numpy as np
import networkx as nx

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

from fpgaconvnet.models.partition import Partition

from fpgaconvnet.platform import Platform

class Network():

    def __init__(self, name, model, graph, batch_size=1, freq=125,
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

        # get the graph and model
        self.model = model
        self.graph = graph

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
        self.platform = Platform()

        # all types of layers
        self.conv_layers = helper.get_all_layers(self.graph, LAYER_TYPE.Convolution)
        self.pool_layers = helper.get_all_layers(self.graph, LAYER_TYPE.Pooling)

        # update partitions
        self.update_partitions()

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
    from fpgaconvnet.models.network.update import update_coarse_in_out_partition

    from fpgaconvnet.models.network.represent import get_model_input_node
    from fpgaconvnet.models.network.represent import get_model_output_node
    from fpgaconvnet.models.network.represent import get_stream_in_coarse
    from fpgaconvnet.models.network.represent import get_stream_out_coarse
    from fpgaconvnet.models.network.represent import get_buffer_depth_in
    from fpgaconvnet.models.network.represent import save_all_partitions

    from fpgaconvnet.models.network.validate import check_ports
    from fpgaconvnet.models.network.validate import check_resources
    # from fpgaconvnet.models.network.validate import get_resources_bad_partitions
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
            latency += partition.get_latency(self.platform.board_freq)
        # return the total latency as well as reconfiguration time
        return latency + (len(partition_list)-1)*self.platform.reconf_time

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

    def load_network(self, network_path):
        # use parser to load configuration
        self.parser.prototxt_to_fpgaconvnet(self, network_path)
        # update partitions
        self.update_partitions()

