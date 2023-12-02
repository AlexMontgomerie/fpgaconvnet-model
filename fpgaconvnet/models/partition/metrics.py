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
    if node is None:
        all_paths = [nx.dag_longest_path(self.graph)]
    else:
         input_node = graphs.get_input_nodes(self.graph)[0]
         # print(input_node)
         all_paths = [max(nx.all_simple_paths(self.graph, input_node, node), key=lambda x: len(x))]

    # initiation interval of the hardware
    interval = self.get_interval()

    def get_path_delay(path):

        # get the hardware model for each node in the path
        node_hw = [ self.graph.nodes[node]["hw"] for node in path ]

        # get the size in
        size_in = [ n.size_in() for n in node_hw ]
        workload_in = [ n.workload_in() for n in node_hw ]

        # get the size out
        size_out = [ n.size_out() for n in node_hw ]
        workload_out = [ n.workload_out() for n in node_hw ]

        # get the latency
        latency = [ n.latency() for n in node_hw ]

        # get the rate in
        rate_in = [ n.rate_in() for n in node_hw ]
        rate_out = [ n.rate_out() for n in node_hw ]

        # get the pipeline depth of each node
        node_depth = [ n.pipeline_depth() for n in node_hw ]
        start_depth = [ n.start_depth() for n in node_hw ]
        pipeline_depth = [ n.pipeline_depth() for n in node_hw ]

        # get the channels in and out
        channels_in = [ n.channels_in() for n in node_hw ]
        channels_out = [ n.channels_out() for n in node_hw ]

        # get the streams in and out
        streams_in = [ n.streams_in() for n in node_hw ]
        # channels_out = [ n.channels_out() for n in node_hw ]

        # streams_in = [ n.streams_in() for n in node_hw ]

        # if len(path) == 1:
        #     return node_depth[0]*node_hw[0].interval()
        # else:
        #     return 0 + get_path_delay(path[1:])


        # old delay calculation
        # delay = sum([ node_depth[j]/rate_in[j] + (interval/size_in[j]) * \
        #         np.prod([ size_in[k]/size_out[k] for k in range(j+1)
        #             ]) for j in range(len(node_hw)) ])
        # multiport delay calculation
        # delay = sum(node_depth) + sum([ (latency[j]/size_in[j]) * \
        #         np.prod([ size_in[k]/size_out[k] for k in range(j+1)
        #             ]) for j in range(len(node_hw)) ])
        # new delay calculation TODO: verify whether we should use latency[j] or interval at the propagation delay stage
        # propagation_delay = sum([(node_depth[j] * latency[j]) / size_in[j] for j in range(len(node_hw))])
        # backpropagation_delay = sum([ (interval/size_in[0]) * \
        #         np.prod([ size_in[k]/size_out[k] for k in range(j+1)
        #             ]) for j in range(len(node_hw)) ])
        # delay = propagation_delay + backpropagation_delay
        curr_node_interval = lambda i: max([ latency[j] for j in range(i) ])
        # curr_rate = lambda i: curr_node_interval(i)/max(size_in[i], size_out[i-1])
        curr_rate = lambda i: curr_node_interval(i)/size_out[i-1]
        # curr_rate = lambda i: min(rate_in[i], rate_out[i-1])
        delay = sum([ node_depth[i]*curr_rate(i) for i in range(1, len(node_hw)) ]) + node_depth[0]
        # delay = sum([ node_depth[i]/curr_rate(i) for i in range(1, len(node_hw)) ]) + node_depth[0]/rate_in[0]
        # delay = sum([ node_depth[i] for i in range(1, len(node_hw)) ]) + node_depth[0]

        # initialise with the first node delay
        delay = node_depth[0]

        # iterate over the nodes in the path
        for i in range(1, len(node_hw)):

            # get how many bursts of the previous node are required
            # to fill the input buffer of the current node
            num_bursts = max(math.ceil(start_depth[i]/channels_in[i]) - 1, 0)

            # get the interval for the previous nodes
            prev_interval = max([ latency[j] for j in range(i) ])

            # get the delay per burst
            delay_per_burst = (prev_interval/workload_in[i]) * channels_in[i]
            # delay_per_burst = channels_out[i-1]

            # add the delay per burst to the total delay
            delay += num_bursts * delay_per_burst

            # add the remaining cycles from the current burst
            delay += start_depth[i] - num_bursts * channels_in[i]

            # add the delay from the pipeline minus the depth filled by the start_depth
            delay += pipeline_depth[i] - start_depth[i]//streams_in[i]
            # delay += pipeline_depth[i]

            # print(node_depth[i], channels_out[i-1], num_bursts, prev_interval, delay_per_burst, delay)

        # append to toal path delays
        print(delay)
        return delay

    # print(all_paths)
    return max([ get_path_delay(path) for path in all_paths ])

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
                workload = hw.workload_in()
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


