import pydot
import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import networkx as nx

class Partition():

    def __init__(
            self,
            graph,
            ports_in=1,
            ports_out=1,
            streams_in=1,
            streams_out=1,
            batch_size=1,
            wr_factor=1,
            port_width=64,
            data_width=16
        ):

        ## graph for partition
        self.graph = graph

        ## batch size
        self.batch_size = batch_size

        ## ports
        self.ports_in   = ports_in
        self.ports_out  = ports_out

        ## streams in and out
        self.streams_in  = [streams_in]
        self.streams_out = [streams_out]

        ## weights reloading
        self.enable_wr  = True
        self.wr_layer   = self.get_wr_layer()
        self.wr_factor  = wr_factor

        ## featuremap size
        self.size_in    = 0
        self.size_out   = 0
        self.size_wr    = 0

        ## bitwidths
        self.port_width     = port_width
        self.data_width     = data_width

        # maximum streams in and out (TODO: turn into function calls)
        self.max_streams_in     = self.ports_in*int(self.port_width/self.data_width)
        self.max_streams_out    = self.ports_out*int(self.port_width/self.data_width)

        self.need_optimise = True

    # auxiliary layer functions
    from fpgaconvnet.models.partition.auxiliary import add_squeeze
    from fpgaconvnet.models.partition.auxiliary import remove_node_by_type
    from fpgaconvnet.models.partition.auxiliary import remove_squeeze

    # metrics
    from fpgaconvnet.models.partition.metrics import get_pipeline_depth
    from fpgaconvnet.models.partition.metrics import get_interval
    from fpgaconvnet.models.partition.metrics import get_cycle
    from fpgaconvnet.models.partition.metrics import get_latency
    from fpgaconvnet.models.partition.metrics import get_total_operations
    from fpgaconvnet.models.partition.metrics import get_total_sparse_operations
    from fpgaconvnet.models.partition.metrics import get_bandwidth_in
    from fpgaconvnet.models.partition.metrics import get_bandwidth_out
    from fpgaconvnet.models.partition.metrics import get_resource_usage

    # update
    from fpgaconvnet.models.partition.update import update
    from fpgaconvnet.models.partition.update import update_multiport_buffer_depth
    from fpgaconvnet.models.partition.update import reduce_squeeze_fanout

    def visualise(self, partition_index):
        cluster = pydot.Cluster(str(partition_index),label=f"partition: {partition_index}",
                spline="ortho", bgcolor="azure", fontsize=25)

        # add mem read and mem write nodes
        cluster.add_node(pydot.Node(f"mem_read_{partition_index}", shape="box",
            style="filled", fillcolor="crimson"))
        cluster.add_node(pydot.Node(f"mem_write_{partition_index}", shape="box",
            style="filled", fillcolor="mediumblue"))

        # get input and output node
        input_node = graphs.get_input_nodes(self.graph)[0]
        output_node = graphs.get_output_nodes(self.graph)[0]

        # add clusters
        edge_labels = {}
        for node in self.graph:
            node_cluster, nodes_in, nodes_out = self.graph.nodes[node]['hw'].visualise(node)
            edge_labels[node] = {
                "nodes_in"  : nodes_in,
                "nodes_out" : nodes_out
            }
            cluster.add_subgraph(node_cluster)
            # add mem read and mem write edges
            if node == input_node:
                for node_in in nodes_in:
                    cluster.add_edge(pydot.Edge(f"mem_read_{partition_index}", node_in))
            if node == output_node:
                for node_out in nodes_out:
                    cluster.add_edge(pydot.Edge(node_out, f"mem_write_{partition_index}"))

        # create edges
        for node in self.graph:
            for edge in graphs.get_next_nodes(self.graph,node):
                for i in range(self.graph.nodes[node]['hw'].streams_out()):
                    cluster.add_edge(pydot.Edge(
                        edge_labels[node]["nodes_out"][i],
                        edge_labels[edge]["nodes_in"][i]))


        _, input_node_vis, _ = self.graph.nodes[input_node]['hw'].visualise(input_node)
        _, _, output_node_vis = self.graph.nodes[output_node]['hw'].visualise(output_node)

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
        input_node  = graphs.get_input_nodes(self.graph)[0]
        max_compute_latency = self.max_compute_node_latency()

        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] == LAYER_TYPE.InnerProduct:
                return False

        return self.graph.nodes[input_node]["type"] == LAYER_TYPE.Squeeze and \
                self.graph.nodes[input_node]["hw"].latency() > max_compute_latency

    def is_output_memory_bound(self):
        output_node  = graphs.get_output_nodes(self.graph)[0]
        max_compute_latency = self.max_compute_node_latency()

        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] == LAYER_TYPE.InnerProduct:
                return False

        return self.graph.nodes[output_node]["type"] == LAYER_TYPE.Squeeze and \
                self.graph.nodes[output_node]["hw"].latency() > max_compute_latency

    def get_wr_layer(self):
        if not self.enable_wr:
            return None
        # all transformable layer types
        transformable_layers = [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]
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
            prev_node = graphs.get_prev_nodes(self.graph,layer)[0]
            return _wr_layer( prev_node )
        # start from the end
        output_node = graphs.get_output_nodes(self.graph)[0]
        if ( self.graph.in_degree(output_node) == 0 ) and \
                ( self.graph.nodes[output_node]['type'] in transformable_layers ):
            return output_node
        else:
            return _wr_layer( output_node )


