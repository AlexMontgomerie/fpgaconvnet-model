import glob
import unittest
import ddt
import copy

from fpgaconvnet.models.network import Network
from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from numpy.linalg import matrix_rank
import scipy
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

PLATFORM = "examples/platforms/zedboard.toml"

class TestNetworkTemplate():

    def run_test_validation(self, network):
        # run all validation checks
        network.check_ports()
        network.check_workload()
        network.check_streams()
        network.check_partitions()
        # network.check_memory_bandwidth()


@ddt.ddt
class TestNetwork(TestNetworkTemplate, unittest.TestCase):

    @ddt.data(*glob.glob("tests/models/*.onnx"))
    def test_network(self, network_path):
        # initialise network
        net = Parser(backend="chisel", quant_mode="auto", convert_gemm_to_conv=False).onnx_to_fpgaconvnet(network_path, save_opt_model=False)

        # load platform
        net.platform.update(PLATFORM)
        # run all tests
        self.run_test_validation(net)
        # self.run_test_partition_transform_split(net)
        # self.run_test_partition_transform_merge(net)

        # test creating a report
        net.create_report("/tmp/report.json")

        # test creating a config file
        net.save_all_partitions("/tmp/config.json")

