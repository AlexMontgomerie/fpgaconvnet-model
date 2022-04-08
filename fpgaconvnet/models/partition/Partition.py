import pydot
import fpgaconvnet_optimiser.tools.graphs as graphs
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

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
            data_width=16,
            weight_width=8,
            acc_width=30,
            port_width=64
        ):

        ## graph for partition
        self.graph = graph

        ## ports
        self.ports_in   = ports_in
        self.ports_out  = ports_out

        ## streams in and out
        self.streams_in  = streams_in
        self.streams_out = streams_out

        ## weights reloading
        self.wr_layer   = self.get_wr_layer()
        self.wr_factor  = wr_factor

        ## featuremap size
        self.size_in    = 0
        self.size_out   = 0
        self.size_wr    = 0

        ## bitwidths
        self.port_width     = port_width
        self.data_width     = data_width
        self.weight_width   = weight_width
        self.acc_width      = acc_width

        # maximum streams in and out (TODO: turn into function calls)
        self.max_streams_in     = self.ports_in*int(self.port_width/self.data_width)
        self.max_streams_out    = self.ports_out*int(self.port_width/self.data_width)

    # metrics
    from fpgaconvnet_optimiser.models.partition.metrics import get_pipeline_depth
    from fpgaconvnet_optimiser.models.partition.metrics import get_interval
    from fpgaconvnet_optimiser.models.partition.metrics import get_latency
    from fpgaconvnet_optimiser.models.partition.metrics import get_total_operations
    from fpgaconvnet_optimiser.models.partition.metrics import get_bandwidth_in
    from fpgaconvnet_optimiser.models.partition.metrics import get_bandwidth_out
    from fpgaconvnet_optimiser.models.partition.metrics import get_resource_usage

    # update
    from fpgaconvnet_optimiser.models.partition.update import update

    def visualise(self, partition_index):
        cluster = pydot.Cluster(str(partition_index),label=f"partition: {partition_index}")
        # add clusters
        edge_labels = {}
        for node in self.graph:
            node_cluster, nodes_in, nodes_out = self.graph.nodes[node]['hw'].visualise(node)
            edge_labels[node] = {
                "nodes_in"  : nodes_in,
                "nodes_out" : nodes_out
            }
            cluster.add_subgraph(node_cluster)
        # create edges
        for node in self.graph:
            for edge in graphs.get_next_nodes(self.graph,node):
                for i in range(self.graph.nodes[node]['hw'].streams_out()):
                    cluster.add_edge(pydot.Edge(edge_labels[node]["nodes_out"][i] ,edge_labels[edge]["nodes_in"][i]))
        # return cluster
        return cluster

    def max_compute_node_latency(self):
        # return max([ self.graph.nodes[node]["hw"].get_latency() for node in
        #              self.graph.nodes() ])
        max_latency = 0
        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] != LAYER_TYPE.Squeeze:
                latency = self.graph.nodes[node]["hw"].get_latency()
                if latency > max_latency:
                    max_latency = latency

        return max_latency

    def is_input_memory_bound(self):
        input_node  = graphs.get_input_nodes(self.graph)[0]
        max_compute_latency = self.max_compute_node_latency()

        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] == LAYER_TYPE.InnerProduct:
                return False

        return self.graph.nodes[input_node]["type"] == LAYER_TYPE.Squeeze and self.graph.nodes[input_node]["hw"].get_latency() > max_compute_latency

    def is_output_memory_bound(self):
        output_node  = graphs.get_output_nodes(self.graph)[0]
        max_compute_latency = self.max_compute_node_latency()

        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] == LAYER_TYPE.InnerProduct:
                return False

        return self.graph.nodes[output_node]["type"] == LAYER_TYPE.Squeeze and self.graph.nodes[output_node]["hw"].get_latency() > max_compute_latency

    def reset(self):
        self.remove_squeeze()
        self.remove_weights_reloading_transform()

        for node in self.graph.nodes():
            self.graph.nodes[node]["hw"].coarse_in = 1
            self.graph.nodes[node]["hw"].coarse_out = 1
            self.graph.nodes[node]["hw"].coarse_group = 1

            if self.graph.nodes[node]["type"] == LAYER_TYPE.Convolution:
                self.graph.nodes[node]["hw"].fine = 1
