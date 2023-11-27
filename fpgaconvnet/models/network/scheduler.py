import numpy as np
import fpgaconvnet.tools.graphs as graphs

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def get_partition_input_dependence(self, partition_index):
    # find input edges
    nodes_in = []
    for input_node in self.partitions[partition_index].input_nodes:
        # get first node that's in graph
        while not input_node in self.graph:
            input_node = graphs.get_next_nodes(self.partitions[partition_index].graph, input_node)[0]
        nodes_in.extend(graphs.get_prev_nodes(self.graph,input_node))
    # find partitions connecting
    partitions_in = []
    for node in nodes_in:
        for i in range(len(self.partitions)):
            # iterate over output nodes
            for output_node in self.partitions[i].output_nodes:
                # find the first output node thats on graph
                while not output_node in self.graph:
                    output_node = graphs.get_prev_nodes(self.partitions[i].graph, output_node)[0]
                if node == output_node:
                    partitions_in.append(i)
    if not partitions_in:
        return [partition_index]
    # return partitions in
    return partitions_in

def get_partition_output_dependence(self, partition_index):
    # find output edges
    nodes_out = []
    for output_node in self.partitions[partition_index].output_nodes:
        # get first node that's in graph
        while not output_node in self.graph:
            output_node = graphs.get_prev_nodes(self.partitions[partition_index].graph, output_node)[0]
        nodes_out.extend(graphs.get_next_nodes(self.graph,output_node))
    # find partitions connecting
    partitions_out = []
    for node in nodes_out:
        for i in range(len(self.partitions)):
            # iterate over input nodes
            for input_node in self.partitions[i].input_nodes:
                # find the first output node thats on graph
                while not input_node in self.graph:
                    input_node = graphs.get_next_nodes(self.partitions[i].graph, input_node)[0]
                if node == input_node:
                    partitions_out.append(i)
    if not partitions_out:
        return [partition_index]
    # return partitions out
    return partitions_out

def get_partition_order(self): # may need to update for
    # find the ordering for the partitions
    partition_order = []
    # function to find which partition a node occurs in
    def _find_node_partition(node):
        ## stores occurence of node in partitions
        partition_occurence = []
        ## iterate over partitions
        for i in range(len(self.partitions)):
            ## check if node exists in partitions graph
            if node in list(self.partitions[i].graph.nodes):
                partition_occurence.append(i)
        ## return first partition node occurs in
        return partition_occurence[0]
    # iterate over each node in graph and find which partition it is in
    nodes = graphs.get_input_nodes(self.graph)
    while nodes:
        # store partition order in this iteration
        partition_order_iter = []
        # get the partition for each node
        for node in nodes:
            partition_order_iter.append(_find_node_partition(node))
        # extend partition order
        partition_order.extend(partition_order_iter)
        # next nodes to be executed
        nodes_next = []
        # iterate over partitions in current iteration
        for i in partition_order_iter:
            for node_out in graphs.get_output_nodes(self.partitions[i].graph):
                while not node_out in self.graph:
                    node_out = graphs.get_prev_nodes(self.partitions[i].graph, node_out)[0]
                for node_next in graphs.get_next_nodes(self.graph, node_out):
                    nodes_next.append(node_next)
        nodes = nodes_next
    # return partition order
    return partition_order

def get_input_base_addr(self, partition_order, partition_index):
    """
    ## check if even partition index
    if (partition_index % 2) == 0:
        ## set base address to 0
        return 0
    ## otherwise set as output of previous partition
    else:
        return _get_output_base_addr(partition_index-1)
    """
    """
    if partition_index == partition_order[0]:
        # start at zero
        return 0
    else:
        # find previous partition
        prev_partition_index = partition_order[partition_order.index(partition_index)-1]
        return self.get_output_base_addr(partition_order, prev_partition_index)
    """
    if partition_index == partition_order[0]:
        # start at zero
        return 0
    else:
        # find the output space of the incoming partition
        return 0 #TODO

def get_output_base_addr(self, partition_order, partition_index):
    """
    ## check if even partition index
    if (partition_index % 2) == 0:
        ## set base address to max between current and next partition
        if (partition_index+1) in partition_order:
            ## get the workload of current partition
            input_node_curr = self.partitions[partition_index]['input_nodes'][0]
            workload_curr   = self.partitions[partition_index]['layers'][input_node_curr]['hw'].workload_in(0)
            ## get the workload of current partition
            output_node_next = self.partitions[partition_index]['output_nodes'][0]
            workload_next    = self.partitions[partition_index]['layers'][output_node_next]['hw'].workload_out(0)
            wr_factor        = self.partitions[partition_index]['wr_factor']
            ## get the max of the workloads
            return max(workload_curr,workload_next*wr_factor) # TODO: *bytes
        else:
            ## get the workload of current partition
            input_node_curr = self.partitions[partition_index]['input_nodes'][0]
            workload_curr   = self.partitions[partition_index]['layers'][input_node_curr]['hw'].workload_in(0)
            return workload_curr # TODO: *bytes
    ## otherwise set as input of previous partition
    else:
        return _get_input_base_addr(partition_index-1)
    """
    return 0

def get_scheduler(self):
    # get the order of partitions
    partition_order = self.get_partition_order()
    # create schedule of partitions
    schedule = []
    for partition_index in partition_order:
        ## input and output nodes
        input_node  = self.partitions[partition_index].input_nodes[0]
        output_node = self.partitions[partition_index].output_nodes[0]
        ## number of ports
        ports_in    = self.partitions[partition_index].ports_in
        ports_out   = self.partitions[partition_index].ports_out
        ports_wr    = 1
        ## input port addresses map
        input_base_addr  = self.get_input_base_addr(partition_order, partition_index)
        output_base_addr = self.get_output_base_addr(partition_order, partition_index)
        ## partition dependencies
        input_dependence    = self.get_partition_input_dependence(partition_index)[0]
        output_dependence   = self.get_partition_output_dependence(partition_index)[0]
        ## dimensions
        batch_size  = self.partitions[partition_index].batch_size
        input_size  = self.partitions[partition_index].graph.nodes[input_node]['hw'].workload_in()
        output_size = self.partitions[partition_index].graph.nodes[output_node]['hw'].workload_out()
        ## weights reloading variables
        wr_factor   = self.partitions[partition_index].wr_factor
        if self.partitions[partition_index].wr_layer:
            ## get size of weights
            wr_layer = self.partitions[partition_index].wr_layer
            weights_size = self.partitions[partition_index].graph.nodes[wr_layer]['hw'].get_parameters_size()['weights']
        else:
            ## set size of weights to zero
            weights_size = 0
        ## append to schedule
        schedule.append({
            'partition_id'              : partition_index,
            'ports_in'                  : ports_in,
            'ports_out'                 : ports_out,
            'input_addr'                : input_base_addr,
            'output_addr'               : output_base_addr,
            'input_dependence'          : input_dependence,
            'output_dependence'         : output_dependence,
            'weights_reloading_factor'  : wr_factor,
            'batch_size'                : batch_size,
            'input_size'                : input_size,
            'output_size'               : output_size,
            'weights_size'              : weights_size
        })
    # return scheduler
    return schedule

def get_schedule_csv(self, output_path):
    # get the schedule
    schedule = self.get_scheduler()
    # setup csv
    n_partitions = len(schedule)
    n_fields     = len(schedule[0].keys())
    csv_out = np.zeros((n_partitions,n_fields),dtype=int)
    # iterate over partitions
    for i in range(n_partitions):
        csv_out[i,0]    = schedule[i]['partition_id']
        csv_out[i,1]    = schedule[i]['ports_in']
        csv_out[i,2]    = schedule[i]['ports_out']
        csv_out[i,3]    = schedule[i]['input_addr']
        csv_out[i,4]    = schedule[i]['output_addr']
        csv_out[i,5]    = schedule[i]['input_dependence']
        csv_out[i,6]    = schedule[i]['output_dependence']
        csv_out[i,7]    = schedule[i]['weights_reloading_factor']
        csv_out[i,8]    = schedule[i]['batch_size']
        csv_out[i,9]    = schedule[i]['input_size']
        csv_out[i,10]   = schedule[i]['output_size']
        csv_out[i,11]   = schedule[i]['weights_size']
    # save csv to output path
    np.savetxt(output_path, csv_out, fmt='%d', delimiter=',')

def check_scheduler(self):
    # get the schedule
    schedule = self.get_scheduler()
    # only check if there's greater than one partition
    if len(schedule) > 1:
        ## check ports
        for i in range(1,len(schedule)):
            assert schedule[i]['ports_in'] == schedule[i-1]['ports_out']
        ## check addresses
        for i in range(1,len(schedule)):
            assert schedule[i]['input_addr'] == schedule[i-1]['output_addr']
    # scheduler fine
    return True

