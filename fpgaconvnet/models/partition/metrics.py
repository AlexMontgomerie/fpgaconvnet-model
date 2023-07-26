import numpy as np
import networkx as nx

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.matrix as matrix

def get_pipeline_depth(self):
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

    # get all the paths between input and output
    all_paths = list(nx.all_simple_paths(self.graph,
        source=graphs.get_input_nodes(self.graph)[0],
        target=graphs.get_output_nodes(self.graph)[-1]))

    path_delays = []

    # # get the longest path
    longest_path = max(all_paths, key=len)
    all_paths = [max(all_paths, key=len)]

    for path in all_paths:

        # get the hardware model for each node in the path
        node_hw = [ self.graph.nodes[node]["hw"] for node in path ]

        # get the size in
        size_in = [ n.size_in() for n in node_hw ]

        # get the size out
        size_out = [ n.size_out() for n in node_hw ]

        # get the latency
        latency = [ n.latency() for n in node_hw ]

        # get the pipeline depth of each node
        node_depth = [ n.pipeline_depth() for n in node_hw ]

        # get the path depth
        delay = sum(node_depth) + sum([ (latency[j]/size_in[j]) * \
                np.prod([ size_in[k]/size_out[k] for k in range(j+1)
                    ]) for j in range(len(node_hw)) ])
        print(delay)
        path_delays.append(delay)

    return max(path_delays)

    # # memoisation of pipeline depths
    # node_pipeline_depth = {}

    # def _pipeline_depth_node(node):

    #     # find the pipeline depth of the current node
    #     pipeline_depth = self.graph.nodes[node]['hw'].pipeline_depth()

    #     # find the longest path to end from this node
    #     if self.graph.out_degree(node) == 0:
    #         return pipeline_depth
    #     elif node in node_pipeline_depth:
    #         return node_pipeline_depth[node]
    #     else:
    #         node_pipeline_depth[node] = pipeline_depth + max([
    #             _pipeline_depth_node(edge) for edge in graphs.get_next_nodes(self.graph, node) ])
    #         return node_pipeline_depth[node]

    # # get the first node of the graph
    # start_node = graphs.get_input_nodes(self.graph)[0]

    # # return pipeline depth from start node
    # return _pipeline_depth_node(start_node)

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
    input_node = graphs.get_input_nodes(self.graph)[0]
    pipeline_depth = self.get_pipeline_depth() # TODO: find max of all input nodes
    # return the latency (in seconds)
    batch_size  = int(self.batch_size)
    wr_factor   = self.wr_factor
    size_wr     = self.size_wr
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
    # get workload and streams in
    bw_in = []
    inputs = graphs.get_input_nodes(self.graph)
    for i, input_node in enumerate(inputs):
        workload = self.graph.nodes[input_node]["hw"].workload_in()
        streams = self.streams_in[i]
        # calculate rate from interval
        rate = workload / (interval*streams)
        # get bandwidth (GB/s)
        # return (rate*streams*self.data_width*freq)/8000
        bw_in.append((rate*streams*self.data_width*freq)/8000)
    return bw_in

def get_bandwidth_out(self,freq):
    # get the interval for the partition
    interval = self.get_interval()
    # get workload and streams out
    bw_out = []
    outputs = graphs.get_output_nodes(self.graph)
    for i, output_node in enumerate(outputs):
        workload = self.graph.nodes[output_node]["hw"].workload_out()
        streams = self.streams_out[i]
        # calculate rate from interval
        rate = workload / (interval*streams)
        # get bandwidth (GB/s)
        # return (rate*streams*self.data_width*freq)/8000
        bw_out.append((rate*streams*self.data_width*freq)/8000)
    return bw_out

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


