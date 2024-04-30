from fpgaconvnet.platform import PlatformBase
from fpgaconvnet.models.network import Network

from fpgaconvnet.models.partition.metrics import get_partition_latency

def get_network_latency(network: Network, platform: PlatformBase) -> float:
    # get the latency of each partition
    partition_latency = [ get_partition_latency(partition, platform) \
                                for partition in network.partitions ]

    # get the latency between partitions
    inter_latency = len(network.partitions)*platform.reconf_time

    # return the total latency
    return sum(partition_latency) + inter_latency

def get_network_cycles(network: Network) -> int:
    return sum([ partition.get_cycles() for partition in network.partitions ])

def get_network_throughput(network: Network, platform: PlatformBase) -> float:
    return float(network.batch_size)/get_network_latency(network, platform)


################################################################################
################################################################################
################################################################################


def get_inter_latency(network, delay, partition_list=None):
    # FIXME: move this to do with platform
    # latency between partitions
    if partition_list == None:
        partition_list = list(range(len(network.partitions)))
    if len(partition_list) == 1:
        return 0
    else:
        return len(partition_list)*delay

def get_cycle(network, pipeline, partition_list=None):
    # FIXME: does this actually make sense at network level? Maybe just remove
    if partition_list == None:
        partition_list = list(range(len(network.partitions)))

    if pipeline:
        # partitions pipelined
        max_interval = 0
        pipeline_depth = 0
        for partition_index, partition in enumerate(network.partitions):
            if partition_index not in partition_list:
                continue
            max_interval = max(max_interval, partition.get_interval())
            pipeline_depth += partition.get_pipeline_depth()
        cycle = int(max_interval*network.batch_size+pipeline_depth)
    else:
        # partitions sequential scheduled
        cycle = 0
        for partition_index, partition in enumerate(network.partitions):
            if partition_index not in partition_list:
                continue
            # accumulate cycle for each partition
            cycle += partition.get_cycle()
    # return the total cycle
    return cycle

# def get_latency(network, freq, pipeline, delay, partition_list=None):
#     if partition_list == None:
#         partition_list = list(range(len(network.partitions)))

#     batch_cycle = network.get_cycle(pipeline, partition_list)
#     latency = batch_cycle/(freq*1000000)
#     # return the total latency
#     return latency + network.get_inter_latency(delay, partition_list)

# def get_throughput(network, freq, pipeline, delay, partition_list=None):
#     if partition_list == None:
#         partition_list = list(range(len(network.partitions)))

#     # return the frames per second
#     return float(network.batch_size)/network.get_latency(freq, pipeline, delay, partition_list)

def get_interval(network, partition_list=None):
    # FIXME: does this actually make sense at network level? Maybe just remove
    assert network.multi_fpga, "get_interval() only works for multi-fpga implementation"
    intervals = []
    for i in partition_list:
        intervals.append(network.partitions[i].get_interval())
    return max(intervals)



