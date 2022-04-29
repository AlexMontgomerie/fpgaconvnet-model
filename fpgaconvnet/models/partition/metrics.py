import numpy as np
import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.matrix as matrix

def get_pipeline_depth(self, node): # TODO: change to longest path problem
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
    # find the pipeline depth of the current node
    pipeline_depth = self.graph.nodes[node]['hw'].pipeline_depth()
    # find the longest path to end from this node
    if self.graph.out_degree(node) == 0:
        return pipeline_depth
    else:
        return pipeline_depth + max([
            self.get_pipeline_depth(edge) for edge in graphs.get_next_nodes(self.graph,node) ])

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
    # get the interval for the partition
    interval = self.get_interval()
    # get pipeline depth of partition
    input_node = graphs.get_input_nodes(self.graph)[0]
    pipeline_depth = self.get_pipeline_depth(input_node) # TODO: find max of all input nodes
    # return the latency (in seconds)
    batch_size  = int(self.batch_size)
    wr_factor   = self.wr_factor
    size_wr     = self.size_wr
    return ( (interval*batch_size+pipeline_depth)*wr_factor + (wr_factor-1)*size_wr )/(frequency*1000000)

def get_bandwidth_in(self,freq):
    # get the interval for the partition
    interval = self.get_interval()
    # get workload and streams in
    input_node = graphs.get_input_nodes(self.graph)[0]
    workload = self.graph.nodes[input_node]["hw"].workload_in()
    streams = self.streams_in
    # calculate rate from interval
    rate = workload / (interval*streams)
    # get bandwidth (GB/s)
    return (rate*streams*self.data_width*freq)/8000

def get_bandwidth_out(self,freq):
    # get the interval for the partition
    interval = self.get_interval()
    # get workload and streams out
    output_node = graphs.get_output_nodes(self.graph)[0]
    workload = self.graph.nodes[output_node]["hw"].workload_out()
    streams = self.streams_out
    # calculate rate from interval
    rate = workload / (interval*streams)
    # get bandwidth (GB/s)
    return (rate*streams*self.data_width*freq)/8000

def get_total_operations(self):
    return sum([self.graph.nodes[node]['hw'].get_operations() for node in self.graph.nodes])

def get_resource_usage(self):
        # initialise resource usage at 0
        resource_usage = { # TODO: initialise with partition resource usage
            'FF'    : 0,
            'LUT'   : 0,
            'DSP'   : 0,
            'BRAM'  : 0
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
        # return resource usage for partition
        return resource_usage


