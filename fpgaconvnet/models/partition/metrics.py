import math
import numpy as np
import networkx as nx

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.matrix as matrix
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.models.layers import MultiPortLayer

def get_initial_output_rates(self, path):

    # dictionary for storing the output rates
    output_rates = {}

    # get the first node's rate out
    output_rates[path[0]] = self.graph.nodes[path[0]]["hw"].rate_out()


    # iterate over the nodes in the path
    for node in path[1:]:

        # get the previous nodes
        prev_nodes = graphs.get_prev_nodes(self.graph, node)

        # get the previous nodes in the output rates
        prev_nodes = [ prev_node for prev_node in prev_nodes if prev_node in output_rates.keys() ]

        # get the rate out based of previous rate out
        match self.graph.nodes[node]["type"]:
            # case LAYER_TYPE.ReSize:
            #     output_rates[node] = min(1, 4*(self.graph.nodes[node]["hw"].rate_out()* \
            #         min(output_rates[prev_nodes[0]] / self.graph.nodes[node]["hw"].rate_in(), 1)))
            # case LAYER_TYPE.Pooling:
            #     output_rates[node] = min(1, 2*(self.graph.nodes[node]["hw"].rate_out()* \
            #         min(output_rates[prev_nodes[0]] / self.graph.nodes[node]["hw"].rate_in(), 1)))
            #         # min(output_rates[prev_nodes[0]], self.graph.nodes[node]["hw"].rate_in())
            case _:
                output_rates[node] = self.graph.nodes[node]["hw"].rate_out() * \
                    min(output_rates[prev_nodes[0]] / self.graph.nodes[node]["hw"].rate_in(), 1)
                    # min(output_rates[prev_nodes[0]], self.graph.nodes[node]["hw"].rate_in())

    # return the dictionary of output rates
    return output_rates

    # # get the previous nodes
    # prev_nodes = nx.predecessors(self.graph, node)

    # if len(prev_nodes) == 0:

    #     return self.graph.nodes[node]["hw"].rate_out()

    # else:

    #     prev_rate_out = self.graph.nodes[prev_nodes[0]]["hw"].rate_out()

    #     return self.graph.nodes[node]["hw"].rate_out() * \
    #         min(prev_rate_out / self.graph.nodes[node]["hw"].rate_in(), 1)


def get_initial_input_rate(self, node):


    # get the previous nodes
    prev_nodes = nx.ancestors(self.graph, node)

    if len(prev_nodes) == 0:

        # return the input rate of the node
        return self.graph.nodes[node]["hw"].rate_in()

    # else:

    #     # get the previous interval of the prior nodes
    #     # prev_intervals = []
    #     # for prev_node in prev_nodes:
    #     #     match self.graph.nodes[prev_node]["type"]:
    #     #         # case LAYER_TYPE.Concat:
    #     #         #     prev_intervals.append(self.graph.nodes[prev_node]["hw"].latency()*2)
    #     #         # case LAYER_TYPE.Pooling:
    #     #         #     prev_intervals.append(self.graph.nodes[prev_node]["hw"].latency()/2)
    #     #         case _:
    #     #             prev_intervals.append(self.graph.nodes[prev_node]["hw"].latency())

    #     # prev_interval = max(prev_intervals)

    #     prev_interval = max([ self.graph.nodes[prev_node]["hw"].latency() for prev_node in prev_nodes ])

    #     # return the input rate based on this previous interval
    #     return self.graph.nodes[node]["hw"].size_in() / prev_interval
    #     # return min(self.graph.nodes[node]["hw"].rate_in(), self.graph.nodes[node]["hw"].size_in() / prev_interval)
    #     # return self.graph.nodes[node]["hw"].rate_in()

    input_node = graphs.get_input_nodes(self.graph)[0]
    path = max(nx.all_simple_paths(
        self.graph, input_node, node), key=lambda x: len(x))

    # get the output rates in the path
    output_rates = self.get_initial_output_rates(path)

    # find the previous nodes
    prev_nodes = graphs.get_prev_nodes(self.graph, node)
    prev_nodes = [ prev_node for prev_node in prev_nodes if prev_node in output_rates.keys() ]

    # get the input rate from the previous output rate
    return output_rates[prev_nodes[0]]

def get_node_delay(self, node):

    # get all the predecessors of the node
    prev_nodes = graphs.get_prev_nodes_all(self.graph, node)
    prev_nodes.append(node)

    # get the subgraph of the predecessors
    path = self.graph.subgraph(prev_nodes)

    # topological sort the subgraph
    path = list(nx.topological_sort(path))

    # get the hardware model for each node in the path
    node_hw = { n: self.graph.nodes[n]["hw"] for n in path }

    # initialise with the first node delay
    delay = node_hw[path[0]].pipeline_depth()
    # delay = 0

    # iterate over the nodes in the path
    for i, node in enumerate(path[1:]):

        # print("\n")

        # get the required words from previous nodes
        required_words = { n: 0 for n in path }
        required_words[path[i]] = node_hw[node].start_depth()

        for j in range(i): # iterate backwards from last node in the graph

            # key for the current node
            curr_node = path[i-j]

            # find all the previous nodes
            prev_nodes = graphs.get_prev_nodes(self.graph, curr_node)

            # get the required_words for each previous node
            for prev_node in prev_nodes:
                required_words[prev_node] += node_hw[curr_node].piecewise_input_words_relationship(required_words[curr_node])

            # prev_node = path[i-j-1]
            # # print(curr_node, prev_node, required_words[curr_node])
            # required_words[prev_node] = node_hw[curr_node].piecewise_input_words_relationship(required_words[curr_node])

        # calculate the output based on the required words
        output_rates = { path[0]: node_hw[path[0]].piecewise_rate_out(1.0, required_words[path[0]]) }
        for j in range(1, i+1): # iterate forwards from the first node in the path

            # key for the current node
            curr_node = path[j]

            # find all the prev nodes
            prev_nodes = graphs.get_prev_nodes(self.graph, curr_node)

            # get the output rate for each next node
            match self.graph.nodes[curr_node]["type"]:
                case LAYER_TYPE.Concat:
                    output_rates[curr_node] = node_hw[curr_node].piecewise_rate_out([output_rates[prev_node] for prev_node in prev_nodes], required_words[curr_node])
                case _:
                    output_rates[curr_node] = node_hw[curr_node].piecewise_rate_out(output_rates[prev_nodes[0]], required_words[curr_node])


            # match self.graph.nodes[curr_node]["type"]:
            #     case LAYER_TYPE.Concat | LAYER_TYPE.EltWise:
            #         prev_node = path[j-1]
            #         output_rates[curr_node] = node_hw[curr_node].piecewise_rate_out([output_rates[prev_node], 1], required_words[curr_node])
            #     case _:
            #         prev_node = path[j-1]
            #         output_rates[curr_node] = node_hw[curr_node].piecewise_rate_out(output_rates[prev_node], required_words[curr_node])

        # using the previous output rate, add to the delay
        delay += node_hw[node].start_depth() / output_rates[path[i]]


    # append to toal path delays
    return delay

def get_node_delay_fast(self, node):

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

        # get the channels per stream
        channels_per_stream = node_hw[i].channels_in() // node_hw[i].streams_in()

        # get how many bursts of the previous node are required
        # to fill the input buffer of the current node
        num_bursts = max(math.ceil(node_hw[i].start_depth()/channels_per_stream) - 1, 0)

        # get the cycles per word
        cycles_per_word = 1 / self.get_initial_input_rate(node)

        # get the delay per burst
        delay_per_burst = cycles_per_word * channels_per_stream

        # add the delay per burst to the total delay
        delay += num_bursts * delay_per_burst

        # add the remaining cycles from the current burst
        delay += (node_hw[i].start_depth() - num_bursts * channels_per_stream) * cycles_per_word

        # add the delay from the pipeline minus the depth filled by the start_depth
        delay += node_hw[i].pipeline_depth() - node_hw[i].start_depth()

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

    # # get the interval for the partition
    # interval = self.get_interval()
    # # get pipeline depth of partition
    # pipeline_depth = self.get_pipeline_depth_fast() # TODO: find max of all input nodes
    # # return the latency (in seconds)
    # batch_size  = int(self.batch_size)
    # wr_factor   = self.wr_factor
    # size_wr     = self.size_wr
    # interval = math.ceil(interval * self.slow_down_factor)
    # batch_cycle = int((interval*batch_size+pipeline_depth)*wr_factor + (wr_factor-1)*size_wr)
    # return batch_cycle

    # calculate the latency for each node, and choose the maximum
    # return max([ self.get_node_delay(node) + self.batch_size*self.graph.nodes[node]['hw'].latency() for node in self.graph.nodes() ])
    batch_cycles = max([ self.get_node_delay(node) + self.batch_size*self.graph.nodes[node]['hw'].latency() for node in self.graph.nodes() ])
    return batch_cycles*self.wr_factor + (self.wr_factor-1)*self.size_wr

    # return batch_cycle

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
                    if self.arch.dimensionality == 2:
                        workload = workload * hw.kernel_size[0]
                    elif self.arch.dimensionality == 3:
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


