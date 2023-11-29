import glob
import unittest
import ddt
import copy
import pytest

from fpgaconvnet.models.network import Network
from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from numpy.linalg import matrix_rank
import scipy
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

PLATFORM = "examples/platforms/zedboard.toml"

ABS_TOL = 100

@ddt.ddt()
def test_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/buffer_test.onnx", save_opt_model=False)

    # set the hardware configuration
    net.partitions[0].graph.nodes["Conv_0"]["hw"].coarse_in = 1
    net.partitions[0].graph.nodes["Conv_0"]["hw"].coarse_out = 2
    net.partitions[0].graph.nodes["Conv_0"]["hw"].fine = 9

    net.partitions[0].graph.nodes["Conv_2"]["hw"].coarse_in = 1
    net.partitions[0].graph.nodes["Conv_2"]["hw"].coarse_out = 16
    net.partitions[0].graph.nodes["Conv_2"]["hw"].fine = 9

    net.partitions[0].graph.nodes["Relu_3"]["hw"].coarse = 2

    net.partitions[0].graph.nodes["GlobalAveragePool_4"]["hw"].coarse = 2

    net.update_partitions()

    # check the correct pipeline depth for each node
    assert net.partitions[0].get_pipeline_depth("Conv_0") == pytest.approx(86, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_1") == pytest.approx(94, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_2") == pytest.approx(520, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_3") == pytest.approx(528, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("GlobalAveragePool_4") == pytest.approx(7050, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_5") == pytest.approx(7317, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_6") == pytest.approx(7320, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_7") == pytest.approx(7415, abs=ABS_TOL)

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

