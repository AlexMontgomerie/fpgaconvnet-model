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

ABS_TOL = 4000

@unittest.skip("Currently disabled in CI.")
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
    net.update_partitions()

    # check the buffer depth estimation for multiport nodes
    assert net.partitions[0].graph.nodes["Concat_11"]["hw"].buffer_depth[0] == pytest.approx(2, abs=ABS_TOL)
    assert net.partitions[0].graph.nodes["Concat_11"]["hw"].buffer_depth[1] == pytest.approx(2720, abs=ABS_TOL)


@unittest.skip("Currently disabled in CI.")
@ddt.ddt()
def test_unet_two_branch_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/unet_two_branch.onnx", save_opt_model=False)

    # set the hardware configuration
    net.partitions[0].graph.nodes["Conv_0"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_0"]["hw"].coarse_out = 1

    net.partitions[0].graph.nodes["Conv_2"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_2"]["hw"].coarse_out = 2

    net.partitions[0].graph.nodes["Conv_5"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_5"]["hw"].coarse_out = 1

    net.partitions[0].graph.nodes["Conv_7"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_7"]["hw"].coarse_out = 2

    net.partitions[0].graph.nodes["Conv_10"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_10"]["hw"].coarse_out = 1

    net.partitions[0].graph.nodes["Conv_12"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_12"]["hw"].coarse_out = 2

    net.partitions[0].graph.nodes["Conv_15"]["hw"].fine = 1
    net.partitions[0].graph.nodes["Conv_15"]["hw"].coarse_out = 4

    net.partitions[0].graph.nodes["Conv_17"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_17"]["hw"].coarse_out = 4

    net.partitions[0].graph.nodes["Conv_19"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_19"]["hw"].coarse_out = 2

    net.partitions[0].graph.nodes["Conv_22"]["hw"].fine = 1
    net.partitions[0].graph.nodes["Conv_22"]["hw"].coarse_out = 4

    net.partitions[0].graph.nodes["Conv_24"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_24"]["hw"].coarse_out = 4

    net.partitions[0].graph.nodes["Conv_26"]["hw"].fine = 9
    net.partitions[0].graph.nodes["Conv_26"]["hw"].coarse_out = 2

    net.partitions[0].graph.nodes["Conv_28"]["hw"].fine = 1
    net.partitions[0].graph.nodes["Conv_28"]["hw"].coarse_out = 1

    net.update_partitions()

    # check the buffer depth estimation for multiport nodes
    assert net.partitions[0].graph.nodes["Concat_16"]["hw"].buffer_depth[0] == pytest.approx(2, abs=ABS_TOL)
    assert net.partitions[0].graph.nodes["Concat_16"]["hw"].buffer_depth[1] == pytest.approx(3200, abs=ABS_TOL)
    # print(net.partitions[0].graph.nodes["Concat_16"]["hw"].buffer_depth)

    assert net.partitions[0].graph.nodes["Concat_23"]["hw"].buffer_depth[0] == pytest.approx(2, abs=ABS_TOL)
    assert net.partitions[0].graph.nodes["Concat_23"]["hw"].buffer_depth[1] == pytest.approx(11550, abs=ABS_TOL)
    # print(net.partitions[0].graph.nodes["Concat_23"]["hw"].buffer_depth)
    # assert False


