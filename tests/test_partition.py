import unittest
import ddt
from fpgaconvnet.models.partition.Partition import Partition
import fpgaconvnet.tools.graphs as graphs
import networkx as nx

from fpgaconvnet.parser import Parser

#class TestPartition(unittest.TestCase):

#    @classmethod
#    def setUpClass(self):
#        # load example network
#        _, graph = parser.parse_net("examples/models/lenet.onnx",view=False)

#        #initialise partition
#        self.partition = Partition(graph)

#    def test_graph(self):
#        # check graph is instance of networkx DiGraph
#        self.assertIsInstance(self.partition.graph, nx.DiGraph)

#        # get input and output nodes
#        input_nodes     = graphs.get_input_nodes(self.partition.graph)
#        output_nodes    = graphs.get_output_nodes(self.partition.graph)

#        # check ports in and out of graph match partition
#        self.assertLessEqual(len(input_nodes) , self.partition.ports_in)
#        self.assertLessEqual(len(output_nodes), self.partition.ports_out)
