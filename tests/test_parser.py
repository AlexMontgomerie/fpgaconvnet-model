import unittest
import ddt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.parser.parser import Parser

# class TestParserLeNet(unittest.TestCase):

#     def setUp(self):
#         self.network_path = "examples/models/lenet.onnx"
#         _, self.graph = parser.parse_net(self.network_path,view=False)

#     def test_graph(self):

#         # check graph
#         pass

#     def test_layer_types(self):

#         # check layer types
#         pass

#     def test_dimensions(self):

#         # check layer dimensions
#         pass

