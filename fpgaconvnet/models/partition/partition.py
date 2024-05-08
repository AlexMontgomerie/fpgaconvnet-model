from dataclasses import dataclass
import math

import pydot
import networkx as nx
import numpy as np

import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE
import fpgaconvnet.parser.onnx.helper as onnx_helper
from fpgaconvnet.architecture import Architecture
import fpgaconvnet.tools.matrix as matrix


@dataclass
class Partition:
    graph: nx.DiGraph
    arch: Architecture
    batch_size: int = 1
    wr_factor: int = 1
    data_width: int = 16

    def __post_init__(self):

        # ports
        self.ports_in = len(graphs.get_input_nodes(
            self.graph, allow_multiport=True))
        self.ports_out = len(graphs.get_output_nodes(
            self.graph, allow_multiport=True))

        # streams in and out
        self.streams_in = [1] * self.ports_in
        self.streams_out = [1] * self.ports_out

        # weights reloading
        self.enable_wr = True
        self.wr_layer = self.get_wr_layer()

        # featuremap size
        self.size_in = 0
        self.size_out = 0
        self.size_wr = 0

        # flag reserved for solver
        self.need_optimise = True
        self.slow_down_factor = 1.0
        self.encode_type = 'none'

    from fpgaconvnet.models.partition.represent import partition_info
    from fpgaconvnet.models.partition.represent import partition_info_dict

    # auxiliary layer functions
    from fpgaconvnet.models.partition.auxiliary import (add_squeeze,
                                                        remove_node_by_type,
                                                        remove_squeeze)

    # metrics
    # from fpgaconvnet.models.partition.metrics import (get_bandwidth_in, get_bandwidth_out,
    #                                                   get_bandwidth_weight, get_cycle, get_interval,
    #                                                   get_latency, get_pipeline_depth, get_resource_usage,
    #                                                   get_total_bandwidth, get_total_operations, get_total_sparse_operations)
    # update
    from fpgaconvnet.models.partition.update import (
        reduce_squeeze_fanout, update, update_multiport_buffer_depth)

    @property
    def input_nodes(self):
        return graphs.get_input_nodes(self.graph, allow_multiport=True)

    @property
    def output_nodes(self):
        return graphs.get_output_nodes(self.graph, allow_multiport=True)

    def visualise(self, partition_index):
        cluster = pydot.Cluster(str(partition_index), label=f"partition: {partition_index}",
                                spline="ortho", bgcolor="azure", fontsize=25)

        # add mem read and mem write nodes
        cluster.add_node(pydot.Node(f"mem_read_{partition_index}", shape="box",
                                    style="filled", fillcolor="crimson"))
        cluster.add_node(pydot.Node(f"mem_write_{partition_index}", shape="box",
                                    style="filled", fillcolor="mediumblue"))

        # get input and output node
        # TODO: fix multiple input/output node support as part of the refactoring of this function
        input_node = graphs.get_input_nodes(self.graph)[0]
        output_node = graphs.get_output_nodes(self.graph)[0]

        # add clusters
        edge_labels = {}
        for node in self.graph:
            node_cluster, nodes_in, nodes_out = self.graph.nodes[node]['hw'].visualise(
                node)
            edge_labels[node] = {
                "nodes_in": nodes_in,
                "nodes_out": nodes_out
            }
            cluster.add_subgraph(node_cluster)
            # add mem read and mem write edges
            if node == input_node:
                for node_in in nodes_in:
                    cluster.add_edge(pydot.Edge(
                        f"mem_read_{partition_index}", node_in))
            if node == output_node:
                for node_out in nodes_out:
                    cluster.add_edge(pydot.Edge(
                        node_out, f"mem_write_{partition_index}"))

        # create edges
        for node in self.graph:
            for edge in graphs.get_next_nodes(self.graph, node):
                for i in range(self.graph.nodes[node]['hw'].streams_out()):
                    cluster.add_edge(pydot.Edge(
                        edge_labels[node]["nodes_out"][i],
                        edge_labels[edge]["nodes_in"][i]))

        _, input_node_vis, _ = self.graph.nodes[input_node]['hw'].visualise(
            input_node)
        _, _, output_node_vis = self.graph.nodes[output_node]['hw'].visualise(
            output_node)

        # return cluster, input_node and output_node
        return cluster, input_node_vis, output_node_vis

    def max_compute_node_latency(self):
        max_latency = 0
        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] != LAYER_TYPE.Squeeze:
                latency = self.graph.nodes[node]["hw"].latency()
                if latency > max_latency:
                    max_latency = latency

        return max_latency

    def is_input_memory_bound(self):
        input_nodes = graphs.get_input_nodes(self.graph, allow_multiport=True)
        max_compute_latency = self.max_compute_node_latency()

        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] == LAYER_TYPE.InnerProduct:
                return False

        return any([self.graph.nodes[input_node]["type"] == LAYER_TYPE.Squeeze and
                    self.graph.nodes[input_node]["hw"].latency() > max_compute_latency for input_node in input_nodes])

    def is_output_memory_bound(self):
        output_nodes = graphs.get_output_nodes(
            self.graph, allow_multiport=True)
        max_compute_latency = self.max_compute_node_latency()

        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] == LAYER_TYPE.InnerProduct:
                return False

        return any([self.graph.nodes[output_node]["type"] == LAYER_TYPE.Squeeze and
                    self.graph.nodes[output_node]["hw"].latency() > max_compute_latency for output_node in output_nodes])

    def is_sparse(self):
        return False
        # TODO: implement this function
        # for node in self.graph.nodes():
        #     if isinstance(self.graph.nodes[node]['hw'], InnerProductSparseLayer) or isinstance(self.graph.nodes[node]['hw'], ConvolutionSparseLayer) or isinstance(self.graph.nodes[node]['hw'], ConvolutionPointwiseSparseLayer):
        #         return True
        # return False

    def get_wr_layer(self):
        if not self.enable_wr:
            return None
        # all transformable layer types
        transformable_layers = [
            LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]
        # iterative function to find weights reloading layer

        def _wr_layer(layer):
            if self.graph.nodes[layer]['type'] == LAYER_TYPE.Split:
                return None
            if self.graph.nodes[layer]['type'] == LAYER_TYPE.Concat:
                return None
            if self.graph.nodes[layer]['type'] == LAYER_TYPE.EltWise:
                return None
            if self.graph.nodes[layer]['type'] in transformable_layers:
                return layer
            if self.graph.in_degree(layer) == 0:
                return None
            prev_node = graphs.get_prev_nodes(self.graph, layer)[0]
            return _wr_layer(prev_node)
        # start from the end
        output_node = graphs.get_output_nodes(self.graph)[0]
        if (self.graph.in_degree(output_node) == 0) and \
                (self.graph.nodes[output_node]['type'] in transformable_layers):
            return output_node
        else:
            return _wr_layer(output_node)

    def check_graph_completeness(self, network_branch_edges):
        """
        Check the validity of the partition with respect to the split and merge layers and their connentivity

        Returns:
            bool: True if the partition is valid, False otherwise
        """
        self.remove_squeeze()

        multiport_layers_out = graphs.get_multiport_layers(self.graph, "out")
        multiport_layers_in = graphs.get_multiport_layers(self.graph, "in")

        if not multiport_layers_out and not multiport_layers_in:
            self.add_squeeze()
            return True, ""

        if (len(multiport_layers_out) > 0 and len(multiport_layers_in) == 0) or (len(multiport_layers_out) == 0 and len(multiport_layers_in) > 0):
            self.add_squeeze()
            return True, ""

        mandatory_edges = []
        for edge in network_branch_edges:
            if edge[0] in self.graph.nodes() and edge[1] in self.graph.nodes():
                mandatory_edges.append(edge)

        for edge in mandatory_edges:
            if edge not in list(self.graph.edges()):
                self.add_squeeze()
                return False, f"Edge {edge} is missing from the partition's graph."

        self.add_squeeze()
        return True, ""

    def get_pipeline_depth(self, node=None):
        """
        Parameters
        ----------
        node: str
            node to calculate the pipeline depth until

        Returns
        -------
        int
            pipeline depth (in cycles) from the first node
            in the partition to `node`
        """

        path_delays = []

        # get the longest path
        all_paths = [nx.dag_longest_path(self.graph)]

        # initiation interval of the hardware
        interval = self.get_interval()

        for path in all_paths:

            # get the hardware model for each node in the path
            node_hw = [ self.graph.nodes[node]["hw"] for node in path ]

            # get the size in
            size_in = [ n.size_in() for n in node_hw ]

            # get the size out
            size_out = [ n.size_out() for n in node_hw ]

            rate_in = [ n.rate_in() for n in node_hw ]

            # get the pipeline depth of each node
            node_depth = [ n.pipeline_depth() for n in node_hw ]

            # get the path depth
            delay = sum([ node_depth[j]/rate_in[j] + (interval/size_in[j]) * \
                    np.prod([ size_in[k]/size_out[k] for k in range(j+1)
                        ]) for j in range(len(node_hw)) ])

            # append to toal path delays
            path_delays.append(delay)

        return max(path_delays)

    def get_pipeline_depth_fast(self):

        # memoisation of pipeline depths
        node_pipeline_depth = {}

        def _pipeline_depth_node(node):

            # find the pipeline depth of the current node
            pipeline_depth = self.graph.nodes[node]['hw'].pipeline_depth()

            # find the longest path to end from this node
            if self.graph.out_degree(node) == 0:
                return pipeline_depth
            elif node in node_pipeline_depth:
                return node_pipeline_depth[node]
            else:
                node_pipeline_depth[node] = pipeline_depth + max([
                    _pipeline_depth_node(edge) for edge in graphs.get_next_nodes(self.graph, node) ])
                return node_pipeline_depth[node]

        # get the first node of the graph
        start_node = graphs.get_input_nodes(self.graph)[0]

        # return pipeline depth from start node
        return _pipeline_depth_node(start_node)

    def get_interval(self):
        """
        Returns
        -------
        int
            gives the interval (in cycles) of the slowest node.
            This is the cycles to process a single featuremap for
            the partition.
        """
        # get the interval matrix
        interval_matrix = matrix.get_interval_matrix(self.graph)
        # return the overall interval
        return np.max(np.absolute(interval_matrix))

    def get_cycles(self):
        # get the interval for the partition
        interval = self.get_interval()
        # get pipeline depth of partition
        input_node = graphs.get_input_nodes(self.graph)[0]
        pipeline_depth = self.get_pipeline_depth() # TODO: find max of all input nodes
        # return the latency (in seconds)
        batch_size  = int(self.batch_size)
        wr_factor   = self.wr_factor
        size_wr     = self.size_wr
        interval = math.ceil(interval * self.slow_down_factor)
        batch_cycle = int((interval*batch_size+pipeline_depth)*wr_factor + (wr_factor-1)*size_wr)
        return batch_cycle

    def get_total_operations(self):
        return sum([self.graph.nodes[node]['hw'].get_operations() for node in self.graph.nodes])

    def get_total_sparse_operations(self):
        return sum([self.graph.nodes[node]['hw'].get_sparse_operations() for node in self.graph.nodes])


