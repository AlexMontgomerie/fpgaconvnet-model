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
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/buffer_test.json")

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

@ddt.ddt()
def test_unet_single_branch_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/unet_single_branch.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/unet_single_branch.json")

    net.update_partitions()

    # check the correct pipeline depth for each node
    assert net.partitions[0].get_pipeline_depth("Conv_0") == pytest.approx(156, abs=ABS_TOL)
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

@ddt.ddt()
def test_unet_two_branch_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/unet_two_branch.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/unet_two_branch.json")

    net.update_partitions()

    # check the correct pipeline depth for each node
    assert net.partitions[0].get_pipeline_depth("Conv_0") == pytest.approx(156, abs=ABS_TOL)
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
def test_unet_three_branch_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/unet_three_branch.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/unet_three_branch.json")

    net.update_partitions()

    # check the correct pipeline depth for each node
    assert net.partitions[0].get_pipeline_depth("Conv_0") == pytest.approx(147, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_1") == pytest.approx(167, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_2") == pytest.approx(1896, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_3") == pytest.approx(1907, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_3_split") == pytest.approx(1913, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("MaxPool_4") == pytest.approx(6152, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_5") == pytest.approx(15100, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_6") == pytest.approx(15103, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_7") == pytest.approx(24323, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_8") == pytest.approx(24334, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_8_split") == pytest.approx(24340, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("MaxPool_9") == pytest.approx(33054, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_10") == pytest.approx(52466, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_11") == pytest.approx(52469, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_12") == pytest.approx(72937, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_13") == pytest.approx(72945, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_13_split") == pytest.approx(72946, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("MaxPool_14") == pytest.approx(91388, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_15") == pytest.approx(136336, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_16") == pytest.approx(136339, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_17") == pytest.approx(185447, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_18") == pytest.approx(185455, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Resize_19") == pytest.approx(185458, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_20") == pytest.approx(187504, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Concat_21") == pytest.approx(187512, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_22") == pytest.approx(220284, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_23") == pytest.approx(220292, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_24") == pytest.approx(240879, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_25") == pytest.approx(240887, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Resize_26") == pytest.approx(240890, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_27") == pytest.approx(241408, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Concat_28") == pytest.approx(241416, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_29") == pytest.approx(257812, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_30") == pytest.approx(257820, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_31") == pytest.approx(267103, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_32") == pytest.approx(267111, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Resize_33") == pytest.approx(267114, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_34") == pytest.approx(267195, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Concat_35") == pytest.approx(267203, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_36") == pytest.approx(275213, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_37") == pytest.approx(275221, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_38") == pytest.approx(277466, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Relu_39") == pytest.approx(277474, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_40") == pytest.approx(277503, abs=ABS_TOL)

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

@ddt.ddt()
def test_resnet8_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/resnet8.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/resnet8.json")

    net.update_partitions()

    # assert net.partitions[0].get_pipeline_depth("Conv_0") == pytest.approx(156, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_2") == pytest.approx(1899, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_4") == pytest.approx(4072, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Add_5") == pytest.approx(4081, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_7") == pytest.approx(4207, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_8") == pytest.approx(8391, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_10") == pytest.approx(12773, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Add_11") == pytest.approx(12782, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_13") == pytest.approx(13289, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_14") == pytest.approx(21634, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Conv_16") == pytest.approx(30882, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Add_17") == pytest.approx(30891, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("GlobalAveragePool_19") == pytest.approx(82828, abs=ABS_TOL)
    assert net.partitions[0].get_pipeline_depth("Gemm_21") == pytest.approx(83472, abs=ABS_TOL)


