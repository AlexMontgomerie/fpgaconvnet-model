import os
import glob
import unittest
import ddt
import copy

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fpgaconvnet.models.network import Network
from fpgaconvnet.parser.parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY

from numpy.linalg import matrix_rank
import scipy
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

PLATFORM = "examples/platforms/zedboard.toml"

ARCHS = [
        Architecture(BACKEND.CHISEL, DIMENSIONALITY.TWO),
        Architecture(BACKEND.CHISEL, DIMENSIONALITY.THREE),
        Architecture(BACKEND.HLS,    DIMENSIONALITY.TWO),
        # Architecture(BACKEND.HLS, DIMENSIONALITY.THREE ),
    ]



class TestNetworkTemplate():

    def run_test_validation(self, network): pass
        # run all validation checks
        # network.check_ports()
        # network.check_workload()
        # network.check_streams()
        # network.check_partitions()
        # network.check_memory_bandwidth()

    def run_test_partition_transform_split(self, network):
        # assume we start from a single parition
        # start by performing the complete partitioning
        network.split_complete()
        # iterate over the partitions
        for partition in network.partitions:
            # check there's only one layer per partition
            self.assertEqual(len(partition.graph.nodes), 1)

    def run_test_partition_transform_merge(self, network):
        # start by splitting the network completely
        network.split_complete()
        # then merge it all together
        network.merge_complete()
        # check that there's only one partition
        self.assertEqual(len(network.partitions), 1)

    def run_test_save_all_partitions(self, network):
        # save all partitions
        network.save_all_partitions("/tmp/fpgaconvnet-test-network-config.json")

@ddt.ddt
class TestNetwork(TestNetworkTemplate, unittest.TestCase):

    @ddt.data(*glob.glob("tests/models/*.onnx"))
    def test_network(self, network_path):
        # initialise network
        net = Parser(backend=BACKEND.CHISEL).onnx_to_fpgaconvnet(network_path, save_opt_model=False)
        # # load platform
        # net.platform.update(PLATFORM)
        # run all tests
        self.run_test_validation(net)
        self.run_test_save_all_partitions(net)
        # self.run_test_partition_transform_split(net)
        # self.run_test_partition_transform_merge(net)

# class TestLoadNetwork(TestNetworkTemplate, unittest.TestCase):

#     def test_single_layer(self):
#         # initialise network
#         net = Network("test", "tests/models/single_layer.onnx")
#         # load the network configuration
#         net.load_network("tests/configs/network/single_layer.json")
#         # run the rest of the tests
#         # load platform
#         net.update_platform(PLATFORM)
#         # run all tests
#         self.run_test_validation(net)
#         self.run_test_partition_transform_split(net)
#         self.run_test_partition_transform_merge(net)

