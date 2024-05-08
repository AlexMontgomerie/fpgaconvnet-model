import math
import numpy as np
import networkx as nx

from .partition import Partition

from fpgaconvnet.platform import PlatformBase
import fpgaconvnet.tools.graphs as graphs
import fpgaconvnet.tools.matrix as matrix
from fpgaconvnet.tools.layer_enum import LAYER_TYPE
from fpgaconvnet.models.layers.metrics import get_layer_resources


def get_partition_latency(partition: Partition, platform: PlatformBase) -> float:
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
    return partition.get_cycles()/(platform.freq*1000000)

def get_partition_bandwidth_in(partition: Partition, platform: PlatformBase) -> list[float]:
    # get the interval for the partition
    interval = partition.get_interval()
    max_latency = interval * partition.slow_down_factor
    # get workload and streams in
    bw_in = []
    inputs = graphs.get_input_nodes(partition.graph)
    for node in partition.graph.nodes():
        hw = partition.graph.nodes[node]["hw"]
        for i in range(len(hw.stream_inputs)):
            if hw.stream_inputs[i] or node in inputs:
                workload = hw.workload_in()
                if partition.graph.nodes[node]["type"] == LAYER_TYPE.Convolution and hw.stream_inputs[i]:
                    # implement line buffer with off-chip memory
                    workload = workload * hw.kernel_size[0]
                streams = hw.streams_in()
                # calculate rate from interval
                rate = workload / (max_latency*streams)
                bitwidth = hw.data_t.width
                # convert bits per cycle to Gbps, freq in MHz
                bw_in.append((rate*streams*bitwidth*platform.freq)/1000)
    return bw_in

def get_partition_bandwidth_out(partition: Partition, platform: PlatformBase) -> list[float]:
    # get the interval for the partition
    interval = partition.get_interval()
    max_latency = interval * partition.slow_down_factor
    # get workload and streams out
    bw_out = []
    outputs = graphs.get_output_nodes(partition.graph)
    for node in partition.graph.nodes():
        hw = partition.graph.nodes[node]["hw"]
        for i in range(len(hw.stream_outputs)):
            if hw.stream_outputs[i] or node in outputs:
                workload = hw.workload_out()
                streams = hw.streams_out()
                # calculate rate from interval
                rate = workload / (max_latency*streams)
                bitwidth = hw.data_t.width
                # convert bits per cycle to Gbps, freq in MHz
                bw_out.append((rate*streams*bitwidth*platform.freq)/1000)
    return bw_out

def get_partition_bandwidth_weight(partition: Partition, platform: PlatformBase) -> list[float]:
    # get the interval for the partition
    interval = partition.get_interval()
    max_latency = interval * partition.slow_down_factor
    # get bandwidth for weights
    bw_weight = []
    for node in partition.graph.nodes():
        if partition.graph.nodes[node]['type'] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
            bits_per_cycle = partition.graph.nodes[node]['hw'].stream_bw()
            latency = partition.graph.nodes[node]['hw'].latency()
            # convert bits per cycle to Gbps, freq in MHz
            bw_weight.append((bits_per_cycle*latency*platform.freq/max_latency)/1000)
    return bw_weight

def get_partition_total_bandwidth(partition: Partition, platform: PlatformBase) -> float:
    bw_in = get_partition_bandwidth_in(partition, platform)
    bw_out = get_partition_bandwidth_out(partition, platform)
    bw_weight = get_partition_bandwidth_weight(partition, platform)
    return sum(bw_in) + sum(bw_out) + sum(bw_weight)

def get_partition_resources(partition: Partition, rsc_type: str, platform: PlatformBase) -> float:
    """
    Get the resource usage of a given partition instance, for the particular
    platform architecture.

    Args:
        partition: the partition to evaluate
        rsc_type: the resource type to evaluate
        platform: the platform to evaluate the module on

    Returns:
        float: the amount of resources
    """
    # initialise the resources as zero
    resources = 0

     # iterate over the nodes of the graph and get the resources
    for layer in partition.graph.nodes():
        resources += get_layer_resources(
                partition.graph.nodes[layer]['hw'],
                rsc_type, platform)

    # return the resources
    return resources

def get_parition_resources_all(partition: Partition, platform: PlatformBase) -> dict[str,float]:

    # iterate over the resource types of the platform , and get the resources
    return { rsc_type: get_partition_resources(partition, rsc_type, platform) \
            for rsc_type in platform.resource_types }


