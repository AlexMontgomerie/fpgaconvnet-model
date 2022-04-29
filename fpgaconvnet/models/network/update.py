import json
import copy

import fpgaconvnet.tools.graphs as graphs

def update_partitions(self):

    # remove all auxiliary layers
    for partition_index in range(len(self.partitions)):

        ## remove squeeze layer
        self.partitions[partition_index].remove_squeeze()

    # remove all empty partitions
    for partition_index in range(len(self.partitions)):

        # remove empty partition
        if len(self.partitions[partition_index].graph.nodes) == 0:
            del self.partitions[partition_index]

    # update partitions
    for partition_index in range(len(self.partitions)):

        ## update the partitions
        self.partitions[partition_index].update()

        ## update batch size for partitions
        self.partitions[partition_index].batch_size = self.batch_size

def update_platform(self, platform_path):

    # get platform
    with open(platform_path,'r') as f:
        platform = json.load(f)

    # update platform information
    #self.platform['name']           = paltform['name']
    self.platform['ports']          = int(platform['ports'])
    #self.platform['port_width']     = int(platform['port_width'])
    #self.platform['freq']           = int(platform['freq'])
    self.platform['reconf_time']    = float(platform['reconf_time'])
    self.platform['mem_capacity']   = int(platform['mem_capacity'])
    self.platform['mem_bandwidth']  = float(platform['mem_bandwidth'])

    # update constraints
    self.platform['constraints']['FF']   = platform['FF']
    self.platform['constraints']['DSP']  = platform['DSP']
    self.platform['constraints']['LUT']  = platform['LUT']
    self.platform['constraints']['BRAM'] = platform['BRAM']

def update_coarse_in_out_partition(self):
    if len(self.partitions) > 1:
        # iterate over partitions
        for i in range(1,len(self.partitions)):
            # get input and output port between partitions
            input_node  = graphs.get_input_nodes(self.partitions[i].graph)[0] # TODO: support multi-port
            output_node = graphs.get_output_nodes(self.partitions[i-1].graph)[0] # TODO: support multi-port
            # update input node's coarse in with previous coarse out
            self.partitions[i].graph.nodes[input_node]['hw'].coarse_in = self.partitions[i-1].graph.nodes[output_node]['hw'].streams_out

