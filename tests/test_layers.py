import unittest
import ddt
import json
from fpgaconvnet.models.layers import *
import glob
import os
from fpgaconvnet.tools.waveform_parser import VCDWaveformParser
import pytest

# Define the path to the hardware backend directory (fpgaconvnet-chisel)
HW_BACKEND_PATH = "../fpgaconvnet-chisel"
ABS_TOL = 100
REL_TOL = 0.025

class TestLayerTemplate():

    def run_test_dimensions(self, layer):
        # check input dimensions
        self.assertTrue(layer.rows_in() > 0)
        self.assertTrue(layer.cols_in() > 0)
        self.assertTrue(layer.channels_in() > 0)
        # check output dimensions
        self.assertTrue(layer.rows_out() > 0)
        self.assertTrue(layer.cols_out() > 0)
        self.assertTrue(layer.channels_out() > 0)

    def run_test_rates(self, layer):
        # check rate in
        self.assertTrue(layer.rate_in() >= 0.0)
        self.assertTrue(layer.rate_in() <= 1.0)
        # check rate out
        self.assertTrue(layer.rate_out() >= 0.0)
        self.assertTrue(layer.rate_out() <= 1.0)

    def run_test_workload(self,layer):
        # check workload in
        self.assertTrue(layer.workload_in() >= 0.0)
        # check workload out
        self.assertTrue(layer.workload_out() >= 0.0)

    def run_test_streams(self,layer):
        # check streams in
        self.assertTrue(layer.streams_in() >= 1)
        # check streams out
        self.assertTrue(layer.streams_out() >= 1)

    def run_test_size(self,layer):
        # check size in
        self.assertTrue(layer.size_in() >= 1)
        # check size out
        self.assertTrue(layer.size_out() >= 1)

    def run_test_latency(self,layer):
        # check latency in
        self.assertTrue(layer.latency_in() >= 0.0)
        # check latency out
        self.assertTrue(layer.latency_out() >= 0.0)

    def run_test_pipeline_depth(self,layer):
        # check pipeline depth
        self.assertTrue(layer.pipeline_depth() >= 0.0)

    def run_test_wait_depth(self,layer):
        # check wait depth
        #self.assertTrue(layer.wait_depth() >= 0.0)
        pass

    def run_test_resources(self,layer):
        # check resources
        rsc = layer.resource()
        for k in rsc:
            self.assertTrue(k in ["BRAM","DSP","LUT","FF","URAM"] )
            self.assertTrue(rsc[k] >= 0)

    def run_test_updating_properties(self, layer):
        # updating coarse in
        coarse_in = max(layer.get_coarse_in_feasible())
        layer.coarse_in = coarse_in
        self.assertEqual(layer.coarse_in, coarse_in)
        # updating coarse out
        coarse_out = max(layer.get_coarse_out_feasible())
        layer.coarse_out = coarse_out
        self.assertEqual(layer.coarse_out, coarse_out)

    def run_hw_simulation(self, layer, index):
        # run hardware simulation
        os.system(f"python {HW_BACKEND_PATH}/scripts/data/generate_layer_data.py -l {layer} -n {index} -p {HW_BACKEND_PATH}")
        # sbt "testOnly fpgaconvnet.layers.${name}_test.ConfigTest"
        os.system(f"cd {HW_BACKEND_PATH} && sbt -Dconfig_idx={index} 'testOnly fpgaconvnet.layers.{layer}_test.ConfigTest' && cd -")

@ddt.ddt
class TestPoolingLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/pooling/config_0.json",
        "tests/configs/layers/pooling/config_1.json",
        "tests/configs/layers/pooling/config_2.json",
        "tests/configs/layers/pooling/config_3.json",
        "tests/configs/layers/pooling/config_4.json",
        "tests/configs/layers/pooling/config_5.json",
        "tests/configs/layers/pooling/config_6.json",
        "tests/configs/layers/pooling/config_7.json",
        "tests/configs/layers/pooling/config_8.json",
        "tests/configs/layers/pooling/config_9.json",
        "tests/configs/layers/pooling/config_10.json",
        "tests/configs/layers/pooling/config_11.json",
        "tests/configs/layers/pooling/config_12.json",
    )
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = PoolingLayer(
            config["rows"],
            config["cols"],
            config["channels"],
            coarse=config["coarse"],
            kernel_rows=config["kernel_size"][0],
            kernel_cols=config["kernel_size"][1],
            stride_rows=config["stride"][0],
            stride_cols=config["stride"][1],
            pad_top=config["pad"],
            pad_left=config["pad"],
            pad_bottom=config["pad"],
            pad_right=config["pad"],
        )

        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)
        self.run_test_workload(layer)
        self.run_test_size(layer)
        self.run_test_streams(layer)
        self.run_test_latency(layer)
        self.run_test_pipeline_depth(layer)
        self.run_test_wait_depth(layer)
        self.run_test_updating_properties(layer)
        self.run_test_resources(layer)

@ddt.ddt
class TestConvolutionLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/convolution/config_0.json",
        # "tests/configs/layers/convolution/config_1.json",
        # "tests/configs/layers/convolution/config_2.json",
        # "tests/configs/layers/convolution/config_3.json",
        # "tests/configs/layers/convolution/config_4.json",
        # "tests/configs/layers/convolution/config_7.json",
        # "tests/configs/layers/convolution/config_8.json",
        # "tests/configs/layers/convolution/config_9.json",
        # "tests/configs/layers/convolution/config_10.json",
        # "tests/configs/layers/convolution/config_11.json",
        # "tests/configs/layers/convolution/config_12.json",
        # "tests/configs/layers/convolution/config_13.json",
        # "tests/configs/layers/convolution/config_14.json",
        # "tests/configs/layers/convolution/config_15.json",
        # "tests/configs/layers/convolution/config_16.json",
        # "tests/configs/layers/convolution/config_17.json",
        # "tests/configs/layers/convolution/config_18.json",
        # "tests/configs/layers/convolution/config_19.json",
         "tests/configs/layers/convolution/config_23.json",
         "tests/configs/layers/convolution/config_25.json",
    )
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = ConvolutionLayer(
            config["filters"],
            config["rows"],
            config["cols"],
            config["channels"],
            coarse_in=config["coarse_in"],
            coarse_out=config["coarse_out"],
            kernel_rows=config["kernel_size"][0],
            kernel_cols=config["kernel_size"][1],
            stride_rows=config["stride"][0],
            stride_cols=config["stride"][1],
            groups=config["groups"],
            pad_top=config["pad"],
            pad_left=config["pad"],
            pad_bottom=config["pad"],
            pad_right=config["pad"],
            fine=config["fine"],
            has_bias=config["has_bias"]
        )

        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)
        self.run_test_workload(layer)
        self.run_test_size(layer)
        self.run_test_streams(layer)
        self.run_test_latency(layer)
        self.run_test_pipeline_depth(layer)
        self.run_test_wait_depth(layer)
        self.run_test_updating_properties(layer)
        self.run_test_resources(layer)

@ddt.ddt
class TestReLULayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/relu/config_0.json",
        "tests/configs/layers/relu/config_1.json",
        "tests/configs/layers/relu/config_2.json",
        "tests/configs/layers/relu/config_3.json",
        "tests/configs/layers/relu/config_4.json",
        "tests/configs/layers/relu/config_5.json",
        "tests/configs/layers/relu/config_6.json",
        "tests/configs/layers/relu/config_7.json",
        "tests/configs/layers/relu/config_8.json",
        "tests/configs/layers/relu/config_9.json",
        "tests/configs/layers/relu/config_10.json",
    )
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = ReLULayer(
            config["rows"],
            config["cols"],
            config["channels"],
            coarse = config["coarse"]
        )

        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)
        self.run_test_workload(layer)
        self.run_test_size(layer)
        self.run_test_streams(layer)
        self.run_test_latency(layer)
        self.run_test_pipeline_depth(layer)
        self.run_test_wait_depth(layer)
        self.run_test_updating_properties(layer)
        self.run_test_resources(layer)

@ddt.ddt
class TestInnerProductLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/inner_product/config_0.json",
        "tests/configs/layers/inner_product/config_1.json",
        "tests/configs/layers/inner_product/config_2.json",
        "tests/configs/layers/inner_product/config_3.json",
        "tests/configs/layers/inner_product/config_4.json",
        "tests/configs/layers/inner_product/config_5.json",
        "tests/configs/layers/inner_product/config_6.json",
        "tests/configs/layers/inner_product/config_7.json",
        "tests/configs/layers/inner_product/config_8.json",
        "tests/configs/layers/inner_product/config_9.json",
        "tests/configs/layers/inner_product/config_10.json",
    )
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = InnerProductLayer(
            config["filters"],
            config["rows"],
            config["cols"],
            config["channels"],
            config["coarse_in"],
            config["coarse_out"],
            has_bias=config["has_bias"]
        )

        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)
        self.run_test_workload(layer)
        self.run_test_size(layer)
        self.run_test_streams(layer)
        self.run_test_latency(layer)
        self.run_test_pipeline_depth(layer)
        self.run_test_wait_depth(layer)
        self.run_test_updating_properties(layer)
        self.run_test_resources(layer)

@ddt.ddt
class TestSqueezeLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/squeeze/config_0.json",
        "tests/configs/layers/squeeze/config_1.json",
    )
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = SqueezeLayer(
            config["rows"],
            config["cols"],
            config["channels"],
            config["coarse_in"],
            config["coarse_out"],
        )

        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)
        self.run_test_workload(layer)
        self.run_test_size(layer)
        self.run_test_streams(layer)
        self.run_test_latency(layer)
        self.run_test_pipeline_depth(layer)
        self.run_test_wait_depth(layer)
        self.run_test_updating_properties(layer)
        self.run_test_resources(layer)

@ddt.ddt
class TestSplitLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/split/config_0.json",
    )
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = SplitLayer(
            config["rows"],
            config["cols"],
            config["channels"],
            config["coarse"],
            ports_out=config["ports_out"]
        )

        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)
        self.run_test_workload(layer)
        self.run_test_size(layer)
        self.run_test_streams(layer)
        self.run_test_latency(layer)
        self.run_test_pipeline_depth(layer)
        self.run_test_wait_depth(layer)
        self.run_test_resources(layer)

@unittest.skip("Currently disabled in CI.")
@ddt.ddt
class TestConvolutionLayer_HW(TestLayerTemplate,unittest.TestCase):

    @ddt.data(*glob.glob(f"{HW_BACKEND_PATH}/data/layers/convolution/test*"))
    def test_layer_configurations(self, test_folder_path):
        test_id = int(test_folder_path.split("/test_")[-1])
        hw_sim_path = f"{HW_BACKEND_PATH}/test_run_dir"

        # List all directories in hw_sim_path
        all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
        # Filter directories based on whether they contain the substring "ConvolutionFixed_Config"
        filtered_dirs = [d for d in all_dirs if "ConvolutionFixed_Config" in d]

        # Check if the specific configuration has an existing simulation run
        found_config = False
        for dir in filtered_dirs:
            if dir.startswith(f'ConvolutionFixed_Config_{test_id}_'):
                found_config = True
                break
        if not found_config:
            self.run_hw_simulation("convolution", test_id)
            # Update filtered_dirs
            all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
            filtered_dirs = [d for d in all_dirs if "ConvolutionFixed_Config" in d]

        # Get the path of the vcd file of the simulation
        for dir in filtered_dirs:
            if dir.startswith(f'ConvolutionFixed_Config_{test_id}_'):
                simulation_dir = dir
                break
        vcd_path = f"{hw_sim_path}/{simulation_dir}/ConvolutionBlockFixed.vcd"
        vcd_parser = VCDWaveformParser(vcd_path)
        simulation_results = vcd_parser.get_layer_stats("Convolution")
        simulation_latency = simulation_results['layer_total_cycles']
        simulation_pipeline_depth = simulation_results['layer_pipeline_depth_cycles']

        config_path = f"{test_folder_path}/config.json"
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = ConvolutionLayer(
            config["filters"],
            config["rows_in"],
            config["cols_in"],
            config["channels_in"],
            coarse_in=config["coarse_in"],
            coarse_out=config["coarse_out"],
            coarse_group=config["coarse_group"],
            kernel_rows=config["kernel_size"][0],
            kernel_cols=config["kernel_size"][1],
            stride_rows=config["stride"][0],
            stride_cols=config["stride"][1],
            groups=config["groups"],
            pad_top=config["pad_top"],
            pad_left=config["pad_left"],
            pad_bottom=config["pad_bottom"],
            pad_right=config["pad_right"],
            fine=config["fine"],
            has_bias=config["bias"],
            input_t=FixedPoint(config["input_t"]["width"], config["input_t"]["binary_point"]),
            weight_t=FixedPoint(config["weight_t"]["width"], config["weight_t"]["binary_point"]),
            output_t=FixedPoint(config["output_t"]["width"], config["output_t"]["binary_point"]),
            acc_t=FixedPoint(config["acc_t"]["width"], config["acc_t"]["binary_point"]),
        )
        layer.update()

        modeling_latency = layer.latency()
        modeling_pipeline_depth = layer.pipeline_depth()

        assert modeling_latency == pytest.approx(simulation_latency, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling latency: {modeling_latency}, simulation latency: {simulation_latency}"
        assert modeling_pipeline_depth == pytest.approx(simulation_pipeline_depth, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling pipeline depth: {modeling_pipeline_depth}, simulation pipeline depth: {simulation_pipeline_depth}"

@unittest.skip("Currently disabled in CI.")
@ddt.ddt
class TestPoolingLayer_HW(TestLayerTemplate,unittest.TestCase):

    @ddt.data(*glob.glob(f"{HW_BACKEND_PATH}/data/layers/pooling/test*"))
    def test_layer_configurations(self, test_folder_path):
        test_id = int(test_folder_path.split("/test_")[-1])
        hw_sim_path = f"{HW_BACKEND_PATH}/test_run_dir"

        # List all directories in hw_sim_path
        all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
        # Filter directories based on whether they contain the substring "PoolingFixed_Config"
        filtered_dirs = [d for d in all_dirs if "PoolingFixed_Config" in d]

        # Check if the specific configuration has an existing simulation run
        found_config = False
        for dir in filtered_dirs:
            if dir.startswith(f'PoolingFixed_Config_{test_id}_'):
                found_config = True
                break
        if not found_config:
            self.run_hw_simulation("pooling", test_id)
            # Update filtered_dirs
            all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
            filtered_dirs = [d for d in all_dirs if "PoolingFixed_Config" in d]

        # Get the path of the vcd file of the simulation
        for dir in filtered_dirs:
            if dir.startswith(f'PoolingFixed_Config_{test_id}_'):
                simulation_dir = dir
                break
        vcd_path = f"{hw_sim_path}/{simulation_dir}/PoolingBlockFixed.vcd"
        vcd_parser = VCDWaveformParser(vcd_path)
        simulation_results = vcd_parser.get_layer_stats("MaxPool")
        simulation_latency = simulation_results['layer_total_cycles']
        simulation_pipeline_depth = simulation_results['layer_pipeline_depth_cycles']

        config_path = f"{test_folder_path}/config.json"
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = PoolingLayer(
            config["rows_in"],
            config["cols_in"],
            config["channels_in"],
            coarse=config["coarse"],
            kernel_rows=config["kernel_size"][0],
            kernel_cols=config["kernel_size"][1],
            stride_rows=config["stride"][0],
            stride_cols=config["stride"][1],
            pad_top=config["pad_top"],
            pad_left=config["pad_left"],
            pad_bottom=config["pad_bottom"],
            pad_right=config["pad_right"],
            data_t=FixedPoint(config["data_t"]["width"], config["data_t"]["binary_point"]),
        )
        modeling_latency = layer.latency()
        modeling_pipeline_depth = layer.pipeline_depth()

        assert modeling_latency == pytest.approx(simulation_latency, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling latency: {modeling_latency}, simulation latency: {simulation_latency}"
        assert modeling_pipeline_depth == pytest.approx(simulation_pipeline_depth, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling pipeline depth: {modeling_pipeline_depth}, simulation pipeline depth: {simulation_pipeline_depth}"

@unittest.skip("Currently disabled in CI.")
@ddt.ddt
class TestGlobalPoolingLayer_HW(TestLayerTemplate,unittest.TestCase):

    @ddt.data(*glob.glob(f"{HW_BACKEND_PATH}/data/layers/average_pooling/test*"))
    def test_layer_configurations(self, test_folder_path):
        test_id = int(test_folder_path.split("/test_")[-1])
        hw_sim_path = f"{HW_BACKEND_PATH}/test_run_dir"

        # List all directories in hw_sim_path
        all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
        # Filter directories based on whether they contain the substring "GlobalAveragePoolingFixed_Config"
        filtered_dirs = [d for d in all_dirs if "GlobalAveragePoolingFixed_Config" in d]

        # Check if the specific configuration has an existing simulation run
        found_config = False
        for dir in filtered_dirs:
            if dir.startswith(f'GlobalAveragePoolingFixed_Config_{test_id}_'):
                found_config = True
                break
        if not found_config:
            self.run_hw_simulation("average_pooling", test_id)
            # Update filtered_dirs
            all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
            filtered_dirs = [d for d in all_dirs if "GlobalAveragePoolingFixed_Config" in d]

        # Get the path of the vcd file of the simulation
        for dir in filtered_dirs:
            if dir.startswith(f'GlobalAveragePoolingFixed_Config_{test_id}_'):
                simulation_dir = dir
                break
        vcd_path = f"{hw_sim_path}/{simulation_dir}/GlobalAveragePoolingFixed.vcd"
        vcd_parser = VCDWaveformParser(vcd_path)
        simulation_results = vcd_parser.get_layer_stats("GlobalAveragePool")
        simulation_latency = simulation_results['layer_total_cycles']
        simulation_pipeline_depth = simulation_results['layer_pipeline_depth_cycles']

        config_path = f"{test_folder_path}/config.json"
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = GlobalPoolingLayer(
            config["rows_in"],
            config["cols_in"],
            config["channels_in"],
            coarse=config["coarse"],
            data_t=FixedPoint(config["data_t"]["width"], config["data_t"]["binary_point"]),
            acc_t=FixedPoint(config["acc_t"]["width"], config["acc_t"]["binary_point"]),
            op_type="avg"
        )
        modeling_latency = layer.latency()
        modeling_pipeline_depth = layer.pipeline_depth()

        assert modeling_latency == pytest.approx(simulation_latency, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling latency: {modeling_latency}, simulation latency: {simulation_latency}"
        assert modeling_pipeline_depth == pytest.approx(simulation_pipeline_depth, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling pipeline depth: {modeling_pipeline_depth}, simulation pipeline depth: {simulation_pipeline_depth}"