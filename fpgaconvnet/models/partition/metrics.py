import math
import numpy as np
import networkx as nx

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.matrix as matrix
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.models.layers import MultiPortLayer

def get_initial_input_rate(self, node):

    # get the previous interval of the prior nodes
    prev_interval = max([ self.graph.nodes[pred]["hw"].latency() for pred in nx.ancestors(self.graph, node) ])

    # return the input rate based on this previous interval
    # return self.graph.nodes[node]["hw"].size_in() / prev_interval
    return min(self.graph.nodes[node]["hw"].rate_in(), self.graph.nodes[node]["hw"].size_in() / prev_interval)

def get_node_delay(self, node):
    def get_total_size_in(n):
        if isinstance(n, MultiPortLayer):
            return sum([ self.graph.nodes[n]["hw"].size_in(j) for j in range(self.graph.nodes[n]["hw"].ports_in) ])
        else:
            return self.graph.nodes[n]["hw"].size_in()

    # get the path to the node
    input_node = graphs.get_input_nodes(self.graph)[0]
    if len(self.graph.nodes()) > 1 and input_node != node:
        path = max(nx.all_simple_paths(
            self.graph, input_node, node), key=lambda x: len(x))
    else:
        path = [input_node]

    # get the hardware model for each node in the path
    node_hw = [ self.graph.nodes[n]["hw"] for n in path ]

    # initialise with the first node delay
    delay = node_hw[0].pipeline_depth()

    # iterate over the nodes in the path
    for i, node in enumerate(path):

        # skip the first node
        if i == 0:
            continue

        # get how many bursts of the previous node are required
        # to fill the input buffer of the current node
        num_bursts = max(math.ceil(node_hw[i].start_depth()/node_hw[i].channels_in()) - 1, 0)

        # get the cycles per word
        cycles_per_word = 1 / self.get_initial_input_rate(node)

        # get the delay per burst
        delay_per_burst = cycles_per_word * node_hw[i].channels_in() // node_hw[i].streams_in()

        # add the delay per burst to the total delay
        delay += num_bursts * delay_per_burst

        # add the remaining cycles from the current burst
        delay += (node_hw[i].start_depth() - num_bursts * node_hw[i].channels_in()) * cycles_per_word

        # add the delay from the pipeline minus the depth filled by the start_depth
        delay += node_hw[i].pipeline_depth() - node_hw[i].start_depth()//node_hw[i].streams_in()

        # print("delay: ", delay, "num_bursts: ", num_bursts, "delay_per_burst: ", delay_per_burst, "cycles_per_word: ", cycles_per_word, "start_depth: ", node_hw[i].start_depth(), "pipeline_depth: ", node_hw[i].pipeline_depth(), "workload_in: ", node_hw[i].workload_in(), "channels_in: ", node_hw[i].channels_in())

    # append to toal path delays
    return delay


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

    # get the longest path
    if node is None:
        output_node = graphs.get_output_nodes(self.graph)[-1]
        return self.get_node_delay(output_node)
    else:
        return self.get_node_delay(node)

    # # find the slowest of all paths
    # return max([ self.get_path_delay(path) for path in all_paths ])

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
    # TODO: fix for multiple input nodes (once this function is actually used)
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

def get_cycle(self):
    # get the interval for the partition
    interval = self.get_interval()
    # get pipeline depth of partition
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
    inputs = graphs.get_input_nodes(self.graph, allow_multiport=True)
    for node in self.graph.nodes():
        hw = self.graph.nodes[node]["hw"]
        for i in range(len(hw.stream_inputs)):
            if hw.stream_inputs[i] or node in inputs:
                workload = hw.workload_in() * hw.input_compression_ratio[i]
                if self.graph.nodes[node]["type"] == LAYER_TYPE.Convolution and hw.stream_inputs[i]:
                    # implement line or tensor buffer with off-chip memory
                    if self.dimensionality == 2:
                        workload = workload * hw.kernel_size[0]
                    elif self.dimensionality == 3:
                        workload = workload * hw.kernel_size[0] * hw.kernel_size[2]
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
    outputs = graphs.get_output_nodes(self.graph, allow_multiport=True)
    for node in self.graph.nodes():
        hw = self.graph.nodes[node]["hw"]
        for i in range(len(hw.stream_outputs)):
            if hw.stream_outputs[i] or node in outputs:
                workload = hw.workload_out() * hw.output_compression_ratio[i]
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
            bits_per_cycle = self.graph.nodes[node]['hw'].stream_bw() \
                * self.graph.nodes[node]['hw'].weight_compression_ratio[0]
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
    ops = 0
    for node in self.graph.nodes():
        if node == self.wr_layer:
            ops += self.graph.nodes[node]['hw'].get_operations() * self.wr_factor
        else:
            ops += self.graph.nodes[node]['hw'].get_operations()
    return ops

def get_total_sparse_operations(self):
    sparse_ops = 0
    for node in self.graph.nodes():
        if node == self.wr_layer:
            sparse_ops += self.graph.nodes[node]['hw'].get_sparse_operations() * self.wr_factor
        else:
            sparse_ops += self.graph.nodes[node]['hw'].get_sparse_operations()
    return sparse_ops

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
        for node in graphs.ordered_node_list(self.graph):
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


