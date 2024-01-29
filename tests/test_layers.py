import unittest
import ddt
import os
import json
import glob
import pytest
import itertools

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.models.exceptions import LayerNotImplementedError, AmbiguousLayerError, ModuleNotImplementedError
from fpgaconvnet.tools.waveform_parser import VCDWaveformParser

# get the paths for all the layer configs
RELU_CONF_PATH=list(glob.glob("tests/configs/layers/relu/*"))
CONVOLUTION_CONF_PATH=list(glob.glob("tests/configs/layers/convolution/*"))
POOLING_CONF_PATH=list(glob.glob("tests/configs/layers/pooling/*"))
INNERPRODUCT_CONF_PATH=list(glob.glob("tests/configs/layers/inner_product/*"))
SQUEEZE_CONF_PATH=list(glob.glob("tests/configs/layers/squeeze/*"))
HARDSWISH_CONF_PATH=list(glob.glob("tests/configs/layers/hardswish/*"))
CONCAT_CONF_PATH=list(glob.glob("tests/configs/layers/concat/*"))
SPLIT_CONF_PATH=list(glob.glob("tests/configs/layers/split/*"))

# get all the architectures
ARCHS = [
        Architecture(BACKEND.CHISEL, DIMENSIONALITY.TWO),
        Architecture(BACKEND.CHISEL, DIMENSIONALITY.THREE),
        Architecture(BACKEND.HLS,    DIMENSIONALITY.TWO),
        # Architecture(BACKEND.HLS, DIMENSIONALITY.THREE ),
    ]

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
        if layer.dimensionality == DIMENSIONALITY.THREE:
            self.assertTrue(layer.depth_in() > 0)
        # check output dimensions
        self.assertTrue(layer.rows_out() > 0)
        self.assertTrue(layer.cols_out() > 0)
        self.assertTrue(layer.channels_out() > 0)
        if layer.dimensionality == DIMENSIONALITY.THREE:
            self.assertTrue(layer.depth_out() > 0)

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
        # check latency
        self.assertTrue(layer.latency() >= 0.0)

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
        # self.assertEqual(set(list(rsc.keys())), set(["URAM", "BRAM","DSP","LUT","FF"]))
        self.assertTrue(rsc["LUT"] >= 0)
        self.assertTrue(rsc["FF"] >= 0)
        # self.assertTrue(rsc["DSP"] >= 0)
        self.assertTrue(rsc["BRAM"] >= 0)
        # self.assertTrue(rsc["URAM"] >= 0)
        # for k in rsc:
        #     self.assertTrue(k in ["BRAM","DSP","LUT","FF","URAM"] )
        #     self.assertTrue(rsc[k] >= 0)

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
# @pytest.mark.skip(reason="Not implemented yet")
class TestPoolingLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*list(itertools.product(ARCHS, POOLING_CONF_PATH)))
    def test_layer_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # get the kernel rows and cols
        config["kernel_rows"] = config["kernel_size"][0]
        config["kernel_cols"] = config["kernel_size"][1]

        # get the stride rows and cols
        config["stride_rows"] = config["stride"][0]
        config["stride_cols"] = config["stride"][1]

        # add dimensionality information
        if arch.dimensionality == DIMENSIONALITY.THREE:
            config["kernel_depth"] = config["kernel_size"][1]
            config["stride_depth"] = config["stride"][1]
            config["depth"] = config["cols"]

        try:
            # initialise layer
            layer = LayerBase.build("pooling", config, arch.backend, arch.dimensionality)

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

        except (LayerNotImplementedError, ModuleNotImplementedError):
            pass

@ddt.ddt
class TestConcatLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*list(itertools.product(ARCHS, CONCAT_CONF_PATH)))
    def test_layer_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # add dimensionality information
        if arch.dimensionality == DIMENSIONALITY.THREE:
            config["depth"] = config["cols"]

        try:
            # initialise layer
            layer = LayerBase.build("concat", config, arch.backend, arch.dimensionality)

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

        except LayerNotImplementedError:
            pass


@ddt.ddt
class TestConvolutionLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*list(itertools.product(ARCHS, CONVOLUTION_CONF_PATH)))
    def test_layer_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # get the kernel rows and cols
        config["kernel_rows"] = config["kernel_size"][0]
        config["kernel_cols"] = config["kernel_size"][1]

        # get the stride rows and cols
        config["stride_rows"] = config["stride"][0]
        config["stride_cols"] = config["stride"][1]

        # add dimensionality information
        if arch.dimensionality == DIMENSIONALITY.THREE:
            config["kernel_depth"] = config["kernel_size"][1]
            config["stride_depth"] = config["stride"][1]
            config["depth"] = config["cols"]

        try:
            # initialise layer
            layer = LayerBase.build("convolution", config, arch.backend, arch.dimensionality)

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

        except (LayerNotImplementedError, ModuleNotImplementedError):
            pass


@ddt.ddt
class TestReLULayer(TestLayerTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*list(itertools.product(ARCHS, RELU_CONF_PATH)))
    def test_layer_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # add dimensionality information
        if arch.dimensionality == DIMENSIONALITY.THREE:
            config["depth"] = config["cols"]

        try:
            # initialise layer
            layer = LayerBase.build("relu", config, arch.backend, arch.dimensionality)

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

        except LayerNotImplementedError:
            pass


# @ddt.ddt
# class TestInnerProductLayer(TestLayerTemplate,unittest.TestCase):

#     @ddt.data(*INNERPRODUCT_CONF_PATH)
#     def test_layer_configurations(self, config_path):

#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise layer
#         layer = Layer.build_from_dict("InnerProductLayer", config)
#         # layer = InnerProductLayer(
#         #     filters=config["filters"],
#         #     rows=config["rows"],
#         #     cols=config["cols"],
#         #     channels=config["channels"],
#         #     coarse_in=config["coarse_in"],
#         #     coarse_out=config["coarse_out"],
#         # )

#         # run tests
#         self.run_test_dimensions(layer)
#         self.run_test_rates(layer)
#         self.run_test_workload(layer)
#         self.run_test_size(layer)
#         self.run_test_streams(layer)
#         self.run_test_latency(layer)
#         self.run_test_pipeline_depth(layer)
#         self.run_test_wait_depth(layer)
#         self.run_test_updating_properties(layer)
#         self.run_test_resources(layer)

@ddt.ddt
class TestSqueezeLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*list(itertools.product(ARCHS, SQUEEZE_CONF_PATH)))
    def test_layer_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # add dimensionality information
        if arch.dimensionality == DIMENSIONALITY.THREE:
            config["depth"] = config["cols"]

        try:
            # initialise layer
            layer = LayerBase.build("squeeze", config, arch.backend, arch.dimensionality)

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

        except LayerNotImplementedError:
            pass


@ddt.ddt
@pytest.mark.skip(reason="Not implemented yet")
class TestHardswishLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(*list(itertools.product(HARDSWISH_CONF_PATH, ARCHS)))
    def test_layer_configurations(self, args):

        # extract the arguments
        config_path, (backend, dimensionality) = args

        # add dimensionality information
        if dimensionality == DIMENSIONALITY.THREE:
            config["depth"] = config["cols"]

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        try:
            # initialise layer
            layer = LayerBase.build("hardswish", config, backend, dimensionality)

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

        except LayerNotImplementedError:
            pass


@ddt.ddt
# @pytest.mark.skip(reason="Not implemented yet")
class TestSplitLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*list(itertools.product(ARCHS, SPLIT_CONF_PATH)))
    def test_layer_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # add dimensionality information
        if arch.dimensionality == DIMENSIONALITY.THREE:
            config["depth"] = config["cols"]

        # initialise layer
        try:
            layer = LayerBase.build("split", config, arch.backend, arch.dimensionality)

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

        except LayerNotImplementedError:
            pass


@pytest.mark.skip
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

@pytest.mark.skip
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


@pytest.mark.skip
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
