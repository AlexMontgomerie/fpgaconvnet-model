
import os
import ddt
import pytest
import unittest
import json

from tabulate import tabulate

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.waveform_parser import VCDWaveformParser

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# Define the path to the hardware backend directory (fpgaconvnet-chisel)
HW_BACKEND_PATH = "../fpgaconvnet-chisel"
ABS_TOL = 2000
REL_TOL = 0.11

def filter_by_type(layer_type):
    return "squeeze" not in layer_type.lower() and "split" not in layer_type.lower() and "reshape" not in layer_type.lower()

@unittest.skip("Currently disabled in CI.")
@ddt.ddt
class TestPipelineDepth(unittest.TestCase):

    @ddt.unpack
    @ddt.data(
        # simple buffer test
        ["tests/models/buffer_test.onnx", "tests/configs/network/buffer_test.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_buffer_test_case_0/PartitionFixedDUT.vcd"],
        # unet single branch
        ["tests/models/unet_single_branch.onnx", "tests/configs/network/unet_single_branch.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_unet_single_branch_case_0/PartitionFixedDUT.vcd"],
        # unet two branch
        ["tests/models/unet_two_branch.onnx", "tests/configs/network/unet_two_branch.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_unet_two_branch_case_0/PartitionFixedDUT.vcd"],
        # unet three branch
        ["tests/models/unet_three_branch.onnx", "tests/configs/network/unet_three_branch.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_unet_three_branch_case_0/PartitionFixedDUT.vcd"],
        # vgg 11 toy
        ["tests/models/vgg11_toy.onnx", "tests/configs/network/vgg11_toy.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_vgg11_toy_case_0/PartitionFixedDUT.vcd"],
        # vgg 11 toy large
        ["tests/models/vgg11_toy_large.onnx", "tests/configs/network/vgg11_toy_large.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_vgg11_toy_large_case_0/PartitionFixedDUT.vcd"],
        # vgg 19 toy
        ["tests/models/vgg19_toy.onnx", "tests/configs/network/vgg19_toy.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_vgg19_toy_case_0/PartitionFixedDUT.vcd"],
        # resnet8
        ["tests/models/resnet8.onnx", "tests/configs/network/resnet8.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_resnet8_case_0/PartitionFixedDUT.vcd"],
        # simple_cnn
        ["tests/models/simple_cnn.onnx", "tests/configs/network/simple_cnn.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_simple_cnn_case_0/PartitionFixedDUT.vcd"],
        # simple_cnn_2
        ["tests/models/simple_cnn_2.onnx", "tests/configs/network/simple_cnn_2.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_simple_cnn_2_case_0/PartitionFixedDUT.vcd"],
        # # yolov5n-320
        ["tests/models/yolov5n-320.onnx", "tests/configs/network/yolov5n-320.json",
            f"{HW_BACKEND_PATH}/test_run_dir/PartitionFixed_Config_0_should_be_correct_for_yolov5n320_case_0/PartitionFixedDUT.vcd"],
    )
    def test_simple_gap_network(self, onnx_path, config_path, vcd_path):

        # initialise network
        parser = Parser(backend="chisel")
        net = parser.onnx_to_fpgaconvnet(onnx_path, save_opt_model=False)
        net = parser.prototxt_to_fpgaconvnet(net, config_path)
        net.update_partitions()

        # find the relavant layers in the model (i.e. not squeeze, split, reshape)
        model_layers = list(filter(filter_by_type, net.partitions[0].graph.nodes()))

        # parse the vcd file
        if os.path.exists(f"{vcd_path}.cache.json"):
            with open(f"{vcd_path}.cache.json", "r") as f:
                partition_stats = json.load(f)
        else:
            vcd_parser = VCDWaveformParser(vcd_path, is_partition=True)
            partition_stats = {}
            for layer in model_layers:
                partition_stats[layer] = vcd_parser.get_layer_stats(layer)
            with open(f"{vcd_path}.cache.json", "w") as f:
                json.dump(partition_stats, f)

        # create a table
        print(tabulate({
            "Layer": [layer for layer in partition_stats],
            "Model Rate In": [net.partitions[0].get_initial_input_rate(layer) for layer in partition_stats],
            "Actual Rate In": [partition_stats[layer]['initial_rate_in_per_stream'] for layer in partition_stats],
            "Actual Rate In\n(avg)": [partition_stats[layer]['average_rate_in_per_stream'] for layer in partition_stats],
            "Model Start Depth\n(words)": [net.partitions[0].graph.nodes[layer]["hw"].start_depth() for layer in partition_stats],
            "Actual Start Depth\n(words)": [partition_stats[layer]['layer_start_depth'] for layer in partition_stats],
            "Model Node Delay\n(cycles)": [net.partitions[0].get_node_delay(layer) for layer in partition_stats],
            "Actual Pipeline Depth\n(cycles)": [partition_stats[layer]['partition_pipeline_depth_cycles'] for layer in partition_stats],
            "Actual latency\n(cycles)": [partition_stats[layer]['last_out_valid_cycles'] - partition_stats[layer]['first_out_valid_cycles'] for layer in partition_stats],
            "Final latency\n(cycles)": [partition_stats[layer]['last_out_valid_cycles'] - 292 for layer in partition_stats],
        }, headers="keys"))

        modeling_latency = net.partitions[0].get_cycle()
        hw_sim_latency = max([partition_stats[layer]['last_out_valid_cycles'] - 292 for layer in partition_stats])
        print()
        print(f"Model  Total latency (cycles): {modeling_latency:.3f}")
        print(f"Actual Total latency (cycles): {hw_sim_latency:.3f}")

        assert modeling_latency == pytest.approx(hw_sim_latency, abs=ABS_TOL, rel=REL_TOL)

        # iterate over layers of the network
        # for layer in partition_stats:
            # check pipeline depth is conservative
            # assert net.partitions[0].get_node_delay(layer) >= partition_stats[layer]['partition_pipeline_depth_cycles']

            # check that it is within a reasonable tolerance
            # assert net.partitions[0].get_node_delay(layer) == pytest.approx(partition_stats[layer]['partition_pipeline_depth_cycles'], abs=ABS_TOL, rel=REL_TOL)


