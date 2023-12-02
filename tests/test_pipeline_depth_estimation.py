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

ABS_TOL = 1000

@ddt.ddt()
def test_simple_gap_network():

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
    # assert net.partitions[0].get_pipeline_depth("Conv_0") == pytest.approx(86, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_1") == pytest.approx(94, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_2") == pytest.approx(520, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_3") == pytest.approx(528, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("GlobalAveragePool_4") == pytest.approx(7050, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_5") == pytest.approx(7317, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_6") == pytest.approx(7320, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_7") == pytest.approx(7415, abs=ABS_TOL)
    # assert False

@ddt.ddt()
def test_unet_single_branch_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/unet_single_branch.onnx", save_opt_model=False)

    # set the hardware configuration
    net.partitions[0].graph.nodes["Conv_0"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_0"]["hw"].coarse_out = 1

    net.partitions[0].graph.nodes["Conv_2"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_2"]["hw"].coarse_out = 2

    net.partitions[0].graph.nodes["Conv_5"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_5"]["hw"].coarse_out = 2

    net.partitions[0].graph.nodes["Conv_7"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_7"]["hw"].coarse_out = 4

    net.partitions[0].graph.nodes["Conv_10"]["hw"].fine = 1
    net.partitions[0].graph.nodes["Conv_10"]["hw"].coarse_out = 8

    net.partitions[0].graph.nodes["Conv_12"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_12"]["hw"].coarse_out = 8

    net.partitions[0].graph.nodes["Conv_14"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_14"]["hw"].coarse_out = 4

    net.partitions[0].graph.nodes["Conv_16"]["hw"].fine = 1
    net.partitions[0].graph.nodes["Conv_16"]["hw"].coarse_out = 1

    net.update_partitions()

    # check the correct pipeline depth for each node
    # assert net.partitions[0].get_pipeline_depth("Conv_0") == pytest.approx(156, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_2") == pytest.approx(1991, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("MaxPool_4") == pytest.approx(6234, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_5") == pytest.approx(14942, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_7") == pytest.approx(23666, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Resize_9") == pytest.approx(23677, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_10") == pytest.approx(23753, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Concat_11") == pytest.approx(23761, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_12") == pytest.approx(27875, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_14") == pytest.approx(32379, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_16") == pytest.approx(32416, abs=ABS_TOL)
    # assert False



