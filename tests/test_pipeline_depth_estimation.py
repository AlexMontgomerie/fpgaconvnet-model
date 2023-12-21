
import os
import ddt
import pytest

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.waveform_parser import VCDWaveformParser

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# Define the path to the hardware backend directory (fpgaconvnet-chisel)
HW_BACKEND_PATH = "../fpgaconvnet-chisel"
ABS_TOL = 500
REL_TOL = 0.05

def filter_by_type(layer_type):
    return "squeeze" not in layer_type.lower() and "split" not in layer_type.lower() and "reshape" not in layer_type.lower()

@ddt.ddt()
def test_simple_gap_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/buffer_test.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/buffer_test.json")
    net.update_partitions()

    model_layers = list(filter(filter_by_type, net.partitions[0].graph.nodes()))

    vcd_path = f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_buffer_test_case_0/PartitionFixedDUT.vcd"
    vcd_parser = VCDWaveformParser(vcd_path, is_partition=True)
    partition_stats = {}
    for layer in model_layers:
        partition_stats[layer] = vcd_parser.get_layer_stats(layer)

    print(f"\nPartition total cycles (modeling estimation): {net.partitions[0].get_cycle()}")

    for layer in partition_stats:
        assert net.partitions[0].get_pipeline_depth(layer) == pytest.approx(partition_stats[layer]['partition_pipeline_depth_cycles'], abs=ABS_TOL, rel=REL_TOL)

@ddt.ddt()
def test_unet_single_branch_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/unet_single_branch.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/unet_single_branch.json")
    net.update_partitions()

    model_layers = list(filter(filter_by_type, net.partitions[0].graph.nodes()))

    vcd_path = f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_unet_single_branch_case_0/PartitionFixedDUT.vcd"
    vcd_parser = VCDWaveformParser(vcd_path, is_partition=True)
    partition_stats = {}
    for layer in model_layers:
        partition_stats[layer] = vcd_parser.get_layer_stats(layer)

    print(f"\nPartition total cycles (modeling estimation): {net.partitions[0].get_cycle()}")

    for layer in partition_stats:
        assert net.partitions[0].get_pipeline_depth(layer) == pytest.approx(partition_stats[layer]['partition_pipeline_depth_cycles'], abs=ABS_TOL, rel=REL_TOL)

@ddt.ddt()
def test_unet_two_branch_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/unet_two_branch.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/unet_two_branch.json")
    net.update_partitions()

    model_layers = list(filter(filter_by_type, net.partitions[0].graph.nodes()))

    vcd_path = f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_unet_two_branch_case_0/PartitionFixedDUT.vcd"
    vcd_parser = VCDWaveformParser(vcd_path, is_partition=True)
    partition_stats = {}
    for layer in model_layers:
        partition_stats[layer] = vcd_parser.get_layer_stats(layer)

    print(f"\nPartition total cycles (modeling estimation): {net.partitions[0].get_cycle()}")

    for layer in partition_stats:
        assert net.partitions[0].get_pipeline_depth(layer) == pytest.approx(partition_stats[layer]['partition_pipeline_depth_cycles'], abs=ABS_TOL, rel=REL_TOL)

@ddt.ddt()
def test_unet_three_branch_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/unet_three_branch.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/unet_three_branch.json")
    net.update_partitions()


    model_layers = list(filter(filter_by_type, net.partitions[0].graph.nodes()))

    vcd_path = f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_unet_three_branch_case_0/PartitionFixedDUT.vcd"
    vcd_parser = VCDWaveformParser(vcd_path, is_partition=True)
    partition_stats = {}
    for layer in model_layers:
        partition_stats[layer] = vcd_parser.get_layer_stats(layer)

    print(f"\nPartition total cycles (modeling estimation): {net.partitions[0].get_cycle()}")

    for layer in partition_stats:
        assert net.partitions[0].get_pipeline_depth(layer) == pytest.approx(partition_stats[layer]['partition_pipeline_depth_cycles'], abs=ABS_TOL, rel=REL_TOL)


@ddt.ddt()
def test_vgg11_toy_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/vgg11_toy.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/vgg11_toy.json")
    net.update_partitions()


    model_layers = list(filter(filter_by_type, net.partitions[0].graph.nodes()))

    vcd_path = f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_vgg11_toy_case_0/PartitionFixedDUT.vcd"
    vcd_parser = VCDWaveformParser(vcd_path, is_partition=True)
    partition_stats = {}
    for layer in model_layers:
        partition_stats[layer] = vcd_parser.get_layer_stats(layer)

    print(f"\nPartition total cycles (modeling estimation): {net.partitions[0].get_cycle()}")

    for layer in partition_stats:
        assert net.partitions[0].get_pipeline_depth(layer) == pytest.approx(partition_stats[layer]['partition_pipeline_depth_cycles'], abs=ABS_TOL, rel=REL_TOL)


@ddt.ddt()
def test_vgg19_toy_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/vgg19_toy.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/vgg19_toy.json")
    net.update_partitions()

    model_layers = list(filter(filter_by_type, net.partitions[0].graph.nodes()))

    vcd_path = f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_vgg19_toy_case_0/PartitionFixedDUT.vcd"
    vcd_parser = VCDWaveformParser(vcd_path, is_partition=True)
    partition_stats = {}
    for layer in model_layers:
        partition_stats[layer] = vcd_parser.get_layer_stats(layer)

    print(f"\nPartition total cycles (modeling estimation): {net.partitions[0].get_cycle()}")

    for layer in partition_stats:
        assert net.partitions[0].get_pipeline_depth(layer) == pytest.approx(partition_stats[layer]['partition_pipeline_depth_cycles'], abs=ABS_TOL, rel=REL_TOL)


@ddt.ddt()
def test_resnet8_network():

    # initialise network
    parser = Parser(backend="chisel")
    net = parser.onnx_to_fpgaconvnet("tests/models/resnet8.onnx", save_opt_model=False)
    net = parser.prototxt_to_fpgaconvnet(net, "tests/configs/network/resnet8.json")
    net.update_partitions()

    model_layers = list(filter(filter_by_type, net.partitions[0].graph.nodes()))

    vcd_path = f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_resnet8_case_0/PartitionFixedDUT.vcd"
    vcd_parser = VCDWaveformParser(vcd_path, is_partition=True)
    partition_stats = {}
    for layer in model_layers:
        partition_stats[layer] = vcd_parser.get_layer_stats(layer)

    print(f"\nPartition total cycles (modeling estimation): {net.partitions[0].get_cycle()}")

    for layer in partition_stats:
        assert net.partitions[0].get_pipeline_depth(layer) == pytest.approx(partition_stats[layer]['partition_pipeline_depth_cycles'], abs=ABS_TOL, rel=REL_TOL)


