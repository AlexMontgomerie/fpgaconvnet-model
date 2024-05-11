import math
import numpy as np
import networkx as nx

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.matrix as matrix
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

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
    interval = max([self.graph.nodes[node]['hw'].latency() for node in self.graph])
    return interval
    
def get_cycle(self):
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

def get_latency(self, frequency):
    """
    Parameters
    ----------
    frequency: float
        board frequency of the FPGA, in MHz.

    Returns
    -------
    int
        the latency of running the partition, in seconds.
    """
    return self.get_cycle()/(frequency*1000000)


def get_bandwidth_in(self,freq):
    # get the interval for the partition
    interval = self.get_interval()
    max_latency = interval * self.slow_down_factor
    # get workload and streams in
    bw_in = []
    inputs = graphs.get_input_nodes(self.graph)
    for node in self.graph.nodes():
        hw = self.graph.nodes[node]["hw"]
        for i in range(len(hw.stream_inputs)):
            if hw.stream_inputs[i] or node in inputs:
                workload = hw.workload_in()
                if self.graph.nodes[node]["type"] == LAYER_TYPE.Convolution and hw.stream_inputs[i]:
                    # implement line buffer with off-chip memory
                    workload = workload * hw.kernel_size[0]
                streams = hw.streams_in()
                # calculate rate from interval
                rate = workload / (max_latency*streams)
                bitwidth = hw.data_t.width
                # convert bits per cycle to Gbps, freq in MHz
                bw_in.append((rate*streams*bitwidth*freq)/1000)
    return bw_in

def get_bandwidth_out(self,freq):
    # get the interval for the partition
    interval = self.get_interval()
    max_latency = interval * self.slow_down_factor
    # get workload and streams out
    bw_out = []
    outputs = graphs.get_output_nodes(self.graph)
    for node in self.graph.nodes():
        hw = self.graph.nodes[node]["hw"]
        for i in range(len(hw.stream_outputs)):
            if hw.stream_outputs[i] or node in outputs:
                workload = hw.workload_out()
                streams = hw.streams_out()
                # calculate rate from interval
                rate = workload / (max_latency*streams)
                bitwidth = hw.data_t.width
                # convert bits per cycle to Gbps, freq in MHz
                bw_out.append((rate*streams*bitwidth*freq)/1000)
    return bw_out

def get_bandwidth_weight(self,freq):
    # get the interval for the partition
    interval = self.get_interval()
    max_latency = interval * self.slow_down_factor
    # get bandwidth for weights
    bw_weight = []
    for node in self.graph.nodes():
        if self.graph.nodes[node]['type'] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
            bits_per_cycle = self.graph.nodes[node]['hw'].stream_bw()
            latency = self.graph.nodes[node]['hw'].latency()
            # convert bits per cycle to Gbps, freq in MHz
            bw_weight.append((bits_per_cycle*latency*freq/max_latency)/1000)
    return bw_weight

def get_total_bandwidth(self,freq):
    bw_in = self.get_bandwidth_in(freq)
    bw_out = self.get_bandwidth_out(freq)
    bw_weight = self.get_bandwidth_weight(freq)
    return sum(bw_in) + sum(bw_out) + sum(bw_weight)

def get_total_operations(self):
    return sum([self.graph.nodes[node]['hw'].get_operations() for node in self.graph.nodes])

def get_total_sparse_operations(self):
    return sum([self.graph.nodes[node]['hw'].get_sparse_operations() for node in self.graph.nodes])

def get_resource_usage(self):
        # initialise resource usage at 0
        resource_usage = { # TODO: initialise with partition resource usage
            'FF'    : 0,
            'LUT'   : 0,
            'DSP'   : 0,
            'BRAM'  : 0,
            'URAM'  : 0
        }
        # iterate over nodes in partition
        for node in self.graph.nodes():
            # get the resource usage of the node
            resource_usage_node = self.graph.nodes[node]['hw'].resource()
            # update total resource usage for partition
            resource_usage['FF']    += resource_usage_node['FF']
            resource_usage['LUT']   += resource_usage_node['LUT']
            resource_usage['DSP']   += resource_usage_node['DSP']
            resource_usage['BRAM']  += resource_usage_node['BRAM']
            if 'URAM' in resource_usage_node:
                resource_usage['URAM']  += resource_usage_node['URAM']
        # return resource usage for partition
        return resource_usage


