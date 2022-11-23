import json
import copy
import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def update(self):

    ## remove auxiliary layers
    self.remove_squeeze()

    ## update streams in and out
    input_node  = graphs.get_input_nodes(self.graph)[0]
    output_node = graphs.get_output_nodes(self.graph)[0]

    ## get valid streams in and out
    streams_in_valid = self.graph.nodes[input_node]["hw"].get_coarse_in_feasible()
    streams_out_valid = self.graph.nodes[output_node]["hw"].get_coarse_out_feasible()

    # get the max stream values in and out
    streams_in_max = min(self.max_streams_in, self.graph.nodes[input_node]["hw"].streams_in())
    streams_out_max = min(self.max_streams_out, self.graph.nodes[output_node]["hw"].streams_out())

    # choose the max of all the valid stream values, below the max
    self.streams_in = max([ s for s in streams_in_valid if s <= streams_in_max ])
    self.streams_out = max([ s for s in streams_out_valid if s <= streams_out_max ])

    ## add auxiliary layers
    self.add_squeeze()

    ## update streams in and out
    self.input_nodes = graphs.get_input_nodes(self.graph)
    self.output_nodes = graphs.get_output_nodes(self.graph)

    ## update sizes
    self.size_in  = self.graph.nodes[self.input_nodes[0]]['hw'].size_in()
    self.size_out = self.graph.nodes[self.input_nodes[0]]['hw'].size_out()
    if self.wr_layer != None:
        self.size_wr = self.graph.nodes[self.wr_layer]['hw'].get_parameters_size()['weights']
    else:
        self.size_wr = 0

    ## update the modules
    for node in self.graph.nodes:
        self.graph.nodes[node]["hw"].update()
