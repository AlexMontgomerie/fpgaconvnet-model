import numpy as np
import copy

import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.matrix as matrix

def check_ports(self):
    # check each partition
    for partition in self.partitions:
        # verify that the number of input ports are not exceeded
        if len(graphs.get_input_nodes(partition.graph)) > self.platform.axi_ports:
            return False
        if len(graphs.get_output_nodes(partition.graph)) > self.platform.axi_ports:
            return False
    return True

def check_resources(self):
    # iterate over partitions
    for partition_index, partition in enumerate(self.partitions):
        # get the resource usage for the platform
        partition_resource_usage = partition.get_resource_usage()
        assert partition_resource_usage['FF']   <= \
                (self.rsc_allocation*self.platform.get_ff()), "ERROR: FF usage exceeded, partition: {partition_index}" 
        assert partition_resource_usage['LUT']  <= \
                (self.rsc_allocation*self.platform.get_lut()), "ERROR: LUT usage exceeded, partition: {partition_index}" 
        assert partition_resource_usage['DSP']  <= \
                (self.rsc_allocation*self.platform.get_dsp()) , "ERROR: DSP usage exceeded, partition: {partition_index}" 
        assert partition_resource_usage['BRAM'] <= \
                (self.rsc_allocation*self.platform.get_bram()), "ERROR: BRAM usage exceeded, partition: {partition_index}" 

def check_workload(self):
    workload_total = np.zeros( shape=( len(self.edge_list),len(self.node_list) ) , dtype=float )
    # iterate over partitions
    for partition_index in range(len(self.partitions)):
        # get weights reloading factor
        wr_factor = self.partitions[partition_index].wr_factor
        # iterate over layers in partition
        for node in self.partitions[partition_index].graph:
            # check layer is in original graph
            if node not in self.graph:
                continue
            # check workload in
            workload_ref = self.graph.nodes[node]['hw'].workload_in()*self.graph.nodes[node]['hw'].streams_in()
            workload_actual = self.partitions[partition_index].graph.nodes[node]['hw'].workload_in()*\
                self.partitions[partition_index].graph.nodes[node]['hw'].streams_in()*wr_factor
            assert workload_actual >= workload_ref, f"({node}) workload in imbalance"
            # check workload out
            workload_ref = self.graph.nodes[node]['hw'].workload_out()*self.graph.nodes[node]['hw'].streams_out()
            workload_actual = self.partitions[partition_index].graph.nodes[node]['hw'].workload_out()*\
                self.partitions[partition_index].graph.nodes[node]['hw'].streams_out()*wr_factor
            assert workload_actual >= workload_ref, f"({node}) workload out imbalance"

def check_streams(self):
    for partition_index in range(len(self.partitions)):
        # get the streams matrix
        streams_matrix = matrix.get_streams_matrix(self.partitions[partition_index].graph)
        # check that the streams cancel
        assert (np.sum(streams_matrix,axis=1) == 0).all(), ""

def check_partitions(self):
    # iterate over partitions
    for p in self.partitions:
        pass

def check_memory_bandwidth(self):
    # iterate over partitions
    for partition in self.partitions:
        # get bandwidth in and out
        bandwidth_in = partition.get_bandwidth_in(self.platform.board_freq)
        bandwidth_out = partition.get_bandwidth_out(self.platform.board_freq)
        # check within platform memory bounds
        assert (bandwidth_in+bandwidth_out) <= self.platform.get_memory_bandwidth(), \
                "Required memory bandwidth is greater than memory bandwidth"
