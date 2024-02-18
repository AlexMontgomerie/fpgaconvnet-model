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


NETWORKS = [ (os.path.basename(network_path), Parser(backend=BACKEND.CHISEL).onnx_to_fpgaconvnet
                (network_path, save_opt_model=False), {}) for network_path \
                        in glob.glob("tests/models/*.onnx") ]

@ddt.ddt
class TestNetwork(unittest.TestCase):

    @ddt.unpack
    @ddt.named_data(*NETWORKS)
    def test_metrics_exist(self, net: Network, config: dict):

        # check the metrics
        assert net.get_latency(200, False, 0) >= 0
        assert net.get_throughput(200, False, 0) >= 0

    @ddt.unpack
    @ddt.named_data(*NETWORKS)
    def test_attributes_exist(self, net: Network, config: dict):

        # check the attributes
        assert hasattr(net, "partitions")
        # run all validation checks
        # network.check_ports()
        # network.check_workload()
        # network.check_streams()
        # network.check_partitions()
        # network.check_memory_bandwidth()


