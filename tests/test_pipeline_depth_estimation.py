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

ABS_TOL = 500

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
    assert False

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

    # check the correct pipeline depth for each node
    # assert net.partitions[0].get_pipeline_depth("Conv_0") == pytest.approx(156, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_2") == pytest.approx(1991, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("MaxPool_4") == pytest.approx(6234, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_5") == pytest.approx(15182, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_7") == pytest.approx(24405, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("MaxPool_9") == pytest.approx(33128, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_10") == pytest.approx(52540, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_12") == pytest.approx(73011, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_15") == pytest.approx(73540, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Concat_16") == pytest.approx(73548, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_17") == pytest.approx(89944, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_19") == pytest.approx(99235, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_22") == pytest.approx(99384, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Concat_23") == pytest.approx(99392, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_24") == pytest.approx(107600, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_26") == pytest.approx(112007, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_28") == pytest.approx(112044, abs=ABS_TOL)


@ddt.ddt()
def test_vgg11_toy_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/vgg11_toy.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/vgg11_toy.json")

    net.update_partitions()

    assert net.partitions[0].get_pipeline_depth("Conv_0") == pytest.approx(139, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_1") == pytest.approx(147, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("MaxPool_2") == pytest.approx(759, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_3") == pytest.approx(2339, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_4") == pytest.approx(2342, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("MaxPool_5") == pytest.approx(8888, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_6") == pytest.approx(23111, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_7") == pytest.approx(23120, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_8") == pytest.approx(41161, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_9") == pytest.approx(41164, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("MaxPool_10") == pytest.approx(82646, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_11") == pytest.approx(167594, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_12") == pytest.approx(167597, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_13") == pytest.approx(254593, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_14") == pytest.approx(254596
, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("MaxPool_15") == pytest.approx(337562, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_16") == pytest.approx(396915, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_17") == pytest.approx(396938, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_18") == pytest.approx(456291, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_19") == pytest.approx(456300, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("GlobalMaxPool_20") == pytest.approx(468597, abs=ABS_TOL)