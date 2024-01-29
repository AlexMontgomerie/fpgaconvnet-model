import glob
import os
import unittest
import ddt
import json
import itertools
import numpy as np
import pytest

from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.models.modules.resources import eval_resource_model
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.models.exceptions import ModuleNotImplementedError
from fpgaconvnet.tools.waveform_parser import VCDWaveformParser

ARCHS = [
        Architecture(BACKEND.CHISEL, DIMENSIONALITY.TWO),
        Architecture(BACKEND.CHISEL, DIMENSIONALITY.THREE),
        Architecture(BACKEND.HLS,    DIMENSIONALITY.TWO),
        # Architecture(BACKEND.HLS, DIMENSIONALITY.THREE ),
    ]

# Define the path to the hardware backend directory (fpgaconvnet-chisel)
HW_BACKEND_PATH = "../fpgaconvnet-chisel"
ABS_TOL = 200
REL_TOL = 0.05
# HW_BACKEND = "chisel"

class TestModuleTemplate():

    def run_test_methods_exist(self, module):
        self.assertTrue(hasattr(module, "rows_in"))
        self.assertTrue(hasattr(module, "cols_in"))
        self.assertTrue(hasattr(module, "channels_in"))
        self.assertTrue(hasattr(module, "rows_out"))
        self.assertTrue(hasattr(module, "cols_out"))
        self.assertTrue(hasattr(module, "channels_out"))

    def run_test_dimensions(self, module):
        # check input dimensions
        self.assertGreater(module.rows_in(), 0)
        self.assertGreater(module.cols_in(), 0)
        self.assertGreater(module.channels_in(), 0)
        # check output dimensions
        self.assertGreater(module.rows_out(), 0)
        self.assertGreater(module.cols_out(), 0)
        self.assertGreater(module.channels_out(), 0)

    def run_test_rates(self, module):
        # check rate in
        for i in range(module.ports_in):
            self.assertGreaterEqual(module.get_rate_in(i), 0.0)
            self.assertLessEqual(module.get_rate_in(i), 1.0)
        # check rate out
        for i in range(module.ports_out):
            self.assertGreaterEqual(module.get_rate_out(i), 0.0)
            self.assertLessEqual(module.get_rate_out(i), 1.0)

    def run_test_latency(self, module):
        self.assertGreaterEqual(module.latency(), 1)

    def run_test_pipeline_depth(self, module):
        self.assertGreaterEqual(module.pipeline_depth(), 0)

    def run_test_resources(self, module):
        self.assertGreaterEqual(eval_resource_model(module, "LUT"), 0)
        self.assertGreaterEqual(eval_resource_model(module, "FF"), 0)
        self.assertGreaterEqual(eval_resource_model(module, "BRAM"), 0)
        # self.assertGreaterEqual(eval_resource_model(module, "DSP"), 0)

    def run_test_config_gen(self, module):

        # generate the config
        config = module.module_info()
        self.assertTrue(isinstance(config, dict))

        # get the module construction parameters
        name = config["name"]
        params = config
        backend = BACKEND[config["backend"]]
        dimensionality = DIMENSIONALITY(config["dimensionality"][0])

        # build from the config
        module = ModuleBase.build(name, params, backend, dimensionality)
        self.assertTrue(isinstance(module, type(module)))

    def run_hw_simulation(self, layer, index):
        # run hardware simulation
        os.system(f"python {HW_BACKEND_PATH}/scripts/data/generate_module_block_data.py -l {layer} -n {index} -p {HW_BACKEND_PATH}")
        os.system(f"cd {HW_BACKEND_PATH} && sbt -Dconfig_idx={index} 'testOnly fpgaconvnet.layers.{layer}_block_test.ConfigTest' && cd -")

@ddt.ddt
class TestForkModule(TestModuleTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/fork/*.json")))
    def test_module_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        config["fine"] = int(np.prod(config["kernel_size"]))

        if isinstance(config["kernel_size"], int):
            config["kernel_size"] = [config["kernel_size"]]*arch.dimensionality.value

        # initialise module
        module = ModuleBase.build("fork", config, arch.backend, arch.dimensionality)

        # run tests
        self.run_test_rates(module)
        self.run_test_latency(module)
        self.run_test_pipeline_depth(module)
        self.run_test_config_gen(module)
        self.run_test_resources(module)

@ddt.ddt
class TestAccumModule(TestModuleTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/accum/*.json")))
    def test_module_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = ModuleBase.build("accum", config, arch.backend, arch.dimensionality)

        # run tests
        self.run_test_rates(module)
        self.run_test_latency(module)
        self.run_test_pipeline_depth(module)
        self.run_test_config_gen(module)
        self.run_test_resources(module)

@ddt.ddt
class TestConvModule(TestModuleTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/conv/*.json")))
    def test_module_configurations(self, arch, config_path):

        if arch.backend == BACKEND.HLS:

            # open configuration
            with open(config_path, "r") as f:
                config = json.load(f)

            if isinstance(config["kernel_size"], int):
                config["kernel_size"] = [config["kernel_size"]]*arch.dimensionality.value

            # initialise module
            module = ModuleBase.build("conv", config, arch.backend, arch.dimensionality)

            # run tests
            self.run_test_rates(module)
            self.run_test_latency(module)
            self.run_test_pipeline_depth(module)
            self.run_test_config_gen(module)
            self.run_test_resources(module)

@ddt.ddt
class TestGlueModule(TestModuleTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/glue/*.json")))
    def test_module_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        config["coarse_group"] = 1
        config["coarse"] = config["coarse_in"]

        # initialise module
        module = ModuleBase.build("glue", config, arch.backend, arch.dimensionality)

        # run tests
        self.run_test_rates(module)
        self.run_test_latency(module)
        self.run_test_pipeline_depth(module)
        self.run_test_config_gen(module)
        self.run_test_resources(module)

@ddt.ddt
class TestSlidingWindowModule(TestModuleTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/sliding_window/*.json")))
    def test_module_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        if isinstance(config["kernel_size"], int):
            config["kernel_size"] = [config["kernel_size"]]*arch.dimensionality.value
        if isinstance(config["stride"], int):
            config["stride"] = [config["stride"]]*arch.dimensionality.value
        if "pad" in config:
            if isinstance(config["pad"], int):
                config["pad"] = [config["pad"]]*arch.dimensionality.value*2
        else:
            config["pad"] = [0]*arch.dimensionality.value*2

        try:
            # initialise module
            module = ModuleBase.build("sliding_window", config, arch.backend, arch.dimensionality)

            # run tests
            self.run_test_rates(module)
            self.run_test_latency(module)
            self.run_test_pipeline_depth(module)
            self.run_test_config_gen(module)
            self.run_test_resources(module)

        except ModuleNotImplementedError:
            pass

@ddt.ddt
class TestPoolModule(TestModuleTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/pool/*.json")))
    def test_module_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # set the pool type
        config["pool_type"] = "max"
        if isinstance(config["kernel_size"], int):
            config["kernel_size"] = [config["kernel_size"]]*arch.dimensionality.value

        try:
            # initialise module
            module = ModuleBase.build("pool", config, arch.backend, arch.dimensionality)

            # run tests
            self.run_test_rates(module)
            self.run_test_latency(module)
            self.run_test_pipeline_depth(module)
            self.run_test_config_gen(module)
            self.run_test_resources(module)

        except ModuleNotImplementedError:
            pass

@ddt.ddt
class TestSqueezeModule(TestModuleTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/squeeze/*.json")))
    def test_module_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = ModuleBase.build("squeeze", config, arch.backend, arch.dimensionality)

        # run tests
        self.run_test_rates(module)
        self.run_test_latency(module)
        self.run_test_pipeline_depth(module)
        self.run_test_config_gen(module)
        self.run_test_resources(module)

@ddt.ddt
class TestReLUModule(TestModuleTemplate,unittest.TestCase):

    @ddt.unpack
    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/relu/*.json")))
    def test_module_configurations(self, arch, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = ModuleBase.build("relu", config, arch.backend, arch.dimensionality)

        # run tests
        self.run_test_rates(module)
        self.run_test_latency(module)
        self.run_test_pipeline_depth(module)
        self.run_test_config_gen(module)
        self.run_test_resources(module)


@pytest.mark.skip
@ddt.ddt
class TestPadModule_HW(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob(f"{HW_BACKEND_PATH}/data/modules/pad_block/test*"))
    def test_module_configurations(self, test_folder_path):
        test_id = int(test_folder_path.split("/test_")[-1])
        hw_sim_path = f"{HW_BACKEND_PATH}/test_run_dir"

        # List all directories in hw_sim_path
        all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
        # Filter directories based on whether they contain the substring "PadBlockFixed_Config"
        filtered_dirs = [d for d in all_dirs if "PadBlockFixed_Config" in d]

        # Check if the specific configuration has an existing simulation run
        found_config = False
        for dir in filtered_dirs:
            if dir.startswith(f'PadBlockFixed_Config_{test_id}_'):
                found_config = True
                break
        if not found_config:
            self.run_hw_simulation("pad", test_id)
            # Update filtered_dirs
            all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
            filtered_dirs = [d for d in all_dirs if "PadBlockFixed_Config" in d]

        # Get the path of the vcd file of the simulation
        for dir in filtered_dirs:
            if dir.startswith(f'PadBlockFixed_Config_{test_id}_'):
                simulation_dir = dir
                break
        vcd_path = f"{hw_sim_path}/{simulation_dir}/PadBlockFixedDUT.vcd"
        vcd_parser = VCDWaveformParser(vcd_path)
        simulation_results = vcd_parser.get_module_stats("Pad")
        simulation_latency = simulation_results['module_total_cycles']
        simulation_pipeline_depth = simulation_results['module_pipeline_depth_cycles']

        config_path = f"{test_folder_path}/config.json"
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        module = Pad(config["rows"],
                     config["cols"],
                     config["channels"],
                     config["pad_top"],
                     config["pad_right"],
                     config["pad_bottom"],
                     config["pad_left"],
                     streams=config["streams"],
                     backend=BACKEND)

        modeling_latency = module.latency()
        modeling_pipeline_depth = module.pipeline_depth()

        assert modeling_latency == pytest.approx(simulation_latency, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling latency: {modeling_latency}, simulation latency: {simulation_latency}"
        assert modeling_pipeline_depth == pytest.approx(simulation_pipeline_depth, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling pipeline depth: {modeling_pipeline_depth}, simulation pipeline depth: {simulation_pipeline_depth}"


@pytest.mark.skip
@ddt.ddt
class TestSlidingWindowModule_HW(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob(f"{HW_BACKEND_PATH}/data/modules/sliding_window_block/test*"))
    def test_module_configurations(self, test_folder_path):
        test_id = int(test_folder_path.split("/test_")[-1])
        hw_sim_path = f"{HW_BACKEND_PATH}/test_run_dir"

        # List all directories in hw_sim_path
        all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
        # Filter directories based on whether they contain the substring "SlidingWindowFixed_Config"
        filtered_dirs = [d for d in all_dirs if "SlidingWindowFixed_Config" in d]

        # Check if the specific configuration has an existing simulation run
        found_config = False
        for dir in filtered_dirs:
            if dir.startswith(f'SlidingWindowFixed_Config_{test_id}_'):
                found_config = True
                break
        if not found_config:
            self.run_hw_simulation("sliding_window", test_id)
            # Update filtered_dirs
            all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
            filtered_dirs = [d for d in all_dirs if "SlidingWindowFixed_Config" in d]

        # Get the path of the vcd file of the simulation
        for dir in filtered_dirs:
            if dir.startswith(f'SlidingWindowFixed_Config_{test_id}_'):
                simulation_dir = dir
                break
        vcd_path = f"{hw_sim_path}/{simulation_dir}/SlidingWindowBlockFixedDUT.vcd"
        vcd_parser = VCDWaveformParser(vcd_path)
        simulation_results = vcd_parser.get_module_stats("SlidingWindow")
        simulation_latency = simulation_results['module_total_cycles']
        simulation_pipeline_depth = simulation_results['module_pipeline_depth_cycles']

        config_path = f"{test_folder_path}/config.json"
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = SlidingWindow(config["rows"],
                               config["cols"],
                               config["channels"],
                               config["kernel_size"],
                               config["stride"],
                               config["pad_top"],
                               config["pad_right"],
                               config["pad_bottom"],
                               config["pad_left"],
                               streams=config["streams"],
                               backend=BACKEND)

        modeling_latency = module.latency()
        modeling_pipeline_depth = module.pipeline_depth()

        assert modeling_latency == pytest.approx(simulation_latency, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling latency: {modeling_latency}, simulation latency: {simulation_latency}"
        assert modeling_pipeline_depth == pytest.approx(simulation_pipeline_depth, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling pipeline depth: {modeling_pipeline_depth}, simulation pipeline depth: {simulation_pipeline_depth}"


@pytest.mark.skip
@ddt.ddt
class TestAccumModule_HW(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob(f"{HW_BACKEND_PATH}/data/modules/accum_block/test*"))
    def test_module_configurations(self, test_folder_path):
        test_id = int(test_folder_path.split("/test_")[-1])
        hw_sim_path = f"{HW_BACKEND_PATH}/test_run_dir"

        # List all directories in hw_sim_path
        all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
        # Filter directories based on whether they contain the substring "AccumBlockFixed_Config"
        filtered_dirs = [d for d in all_dirs if "AccumBlockFixed_Config" in d]

        # Check if the specific configuration has an existing simulation run
        found_config = False
        for dir in filtered_dirs:
            if dir.startswith(f'AccumBlockFixed_Config_{test_id}_'):
                found_config = True
                break
        if not found_config:
            self.run_hw_simulation("accum", test_id)
            # Update filtered_dirs
            all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
            filtered_dirs = [d for d in all_dirs if "AccumBlockFixed_Config" in d]

        # Get the path of the vcd file of the simulation
        for dir in filtered_dirs:
            if dir.startswith(f'AccumBlockFixed_Config_{test_id}_'):
                simulation_dir = dir
                break
        vcd_path = f"{hw_sim_path}/{simulation_dir}/AccumBlockFixedDUT.vcd"
        vcd_parser = VCDWaveformParser(vcd_path)
        simulation_results = vcd_parser.get_module_stats("Accum")
        simulation_latency = simulation_results['module_total_cycles']
        simulation_pipeline_depth = simulation_results['module_pipeline_depth_cycles']

        config_path = f"{test_folder_path}/config.json"
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Accum(config["rows"],
                       config["cols"],
                       config["channels"],
                       config["filters"],
                       config["groups"],
                       streams=config["streams"],
                       backend=BACKEND)

        modeling_latency = module.latency()
        modeling_pipeline_depth = module.pipeline_depth()

        assert modeling_latency == pytest.approx(simulation_latency, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling latency: {modeling_latency}, simulation latency: {simulation_latency}"
        assert modeling_pipeline_depth == pytest.approx(simulation_pipeline_depth, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling pipeline depth: {modeling_pipeline_depth}, simulation pipeline depth: {simulation_pipeline_depth}"


@pytest.mark.skip
@ddt.ddt
class TestSqueezeModule_HW(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob(f"{HW_BACKEND_PATH}/data/modules/squeeze_block/test*"))
    def test_module_configurations(self, test_folder_path):
        test_id = int(test_folder_path.split("/test_")[-1])
        hw_sim_path = f"{HW_BACKEND_PATH}/test_run_dir"

        # List all directories in hw_sim_path
        all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
        # Filter directories based on whether they contain the substring "SqueezeBlockFixed_Config"
        filtered_dirs = [d for d in all_dirs if "SqueezeBlockFixed_Config" in d]

        # Check if the specific configuration has an existing simulation run
        found_config = False
        for dir in filtered_dirs:
            if dir.startswith(f'SqueezeBlockFixed_Config_{test_id}_'):
                found_config = True
                break
        if not found_config:
            self.run_hw_simulation("squeeze", test_id)
            # Update filtered_dirs
            all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
            filtered_dirs = [d for d in all_dirs if "SqueezeBlockFixed_Config" in d]

        # Get the path of the vcd file of the simulation
        for dir in filtered_dirs:
            if dir.startswith(f'SqueezeBlockFixed_Config_{test_id}_'):
                simulation_dir = dir
                break
        vcd_path = f"{hw_sim_path}/{simulation_dir}/SqueezeBlockFixedDUT.vcd"
        vcd_parser = VCDWaveformParser(vcd_path)
        simulation_results = vcd_parser.get_module_stats("Squeeze")
        simulation_latency = simulation_results['module_total_cycles']
        simulation_pipeline_depth = simulation_results['module_pipeline_depth_cycles']

        config_path = f"{test_folder_path}/config.json"
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Squeeze(config["rows"],
                         config["cols"],
                         config["channels"],
                         config["coarse_in"],
                         config["coarse_out"],
                         streams=config["streams"],
                         backend=BACKEND)

        modeling_latency = module.latency()
        modeling_pipeline_depth = module.pipeline_depth()

        assert modeling_latency == pytest.approx(simulation_latency, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling latency: {modeling_latency}, simulation latency: {simulation_latency}"
        assert modeling_pipeline_depth == pytest.approx(simulation_pipeline_depth, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling pipeline depth: {modeling_pipeline_depth}, simulation pipeline depth: {simulation_pipeline_depth}"


@pytest.mark.skip
@ddt.ddt
class TestVectorDotModule_HW(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob(f"{HW_BACKEND_PATH}/data/modules/vector_dot_block/test*"))
    def test_module_configurations(self, test_folder_path):
        test_id = int(test_folder_path.split("/test_")[-1])
        hw_sim_path = f"{HW_BACKEND_PATH}/test_run_dir"

        # List all directories in hw_sim_path
        all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
        # Filter directories based on whether they contain the substring "VectorDotBlockFixed_Config"
        filtered_dirs = [d for d in all_dirs if "VectorDotBlockFixed_Config" in d]

        # Check if the specific configuration has an existing simulation run
        found_config = False
        for dir in filtered_dirs:
            if dir.startswith(f'VectorDotBlockFixed_Config_{test_id}_'):
                found_config = True
                break
        if not found_config:
            self.run_hw_simulation("vector_dot", test_id)
            # Update filtered_dirs
            all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
            filtered_dirs = [d for d in all_dirs if "VectorDotBlockFixed_Config" in d]

        # Get the path of the vcd file of the simulation
        for dir in filtered_dirs:
            if dir.startswith(f'VectorDotBlockFixed_Config_{test_id}_'):
                simulation_dir = dir
                break
        vcd_path = f"{hw_sim_path}/{simulation_dir}/VectorDotBlockFixedDUT.vcd"
        vcd_parser = VCDWaveformParser(vcd_path)
        simulation_results = vcd_parser.get_module_stats("VectorDot")
        simulation_latency = simulation_results['module_total_cycles']
        simulation_pipeline_depth = simulation_results['module_pipeline_depth_cycles']

        config_path = f"{test_folder_path}/config.json"
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = VectorDot(config["rows"],
                           config["cols"],
                           config["channels"],
                           config["filters"],
                           config["fine"],
                           streams=config["streams"],
                           backend=BACKEND)

        modeling_latency = module.latency()
        modeling_pipeline_depth = module.pipeline_depth()

        assert modeling_latency == pytest.approx(simulation_latency, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling latency: {modeling_latency}, simulation latency: {simulation_latency}"
        assert modeling_pipeline_depth == pytest.approx(simulation_pipeline_depth, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling pipeline depth: {modeling_pipeline_depth}, simulation pipeline depth: {simulation_pipeline_depth}"


@pytest.mark.skip
@ddt.ddt
class TestBiasModule_HW(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob(f"{HW_BACKEND_PATH}/data/modules/bias_block/test*"))
    def test_module_configurations(self, test_folder_path):
        test_id = int(test_folder_path.split("/test_")[-1])
        hw_sim_path = f"{HW_BACKEND_PATH}/test_run_dir"

        # List all directories in hw_sim_path
        all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
        # Filter directories based on whether they contain the substring "BiasBlockFixed_Config"
        filtered_dirs = [d for d in all_dirs if "BiasBlockFixed_Config" in d]

        # Check if the specific configuration has an existing simulation run
        found_config = False
        for dir in filtered_dirs:
            if dir.startswith(f'BiasBlockFixed_Config_{test_id}_'):
                found_config = True
                break
        if not found_config:
            self.run_hw_simulation("bias", test_id)
            # Update filtered_dirs
            all_dirs = [d for d in os.listdir(hw_sim_path) if os.path.isdir(os.path.join(hw_sim_path, d))]
            filtered_dirs = [d for d in all_dirs if "BiasBlockFixed_Config" in d]

        # Get the path of the vcd file of the simulation
        for dir in filtered_dirs:
            if dir.startswith(f'BiasBlockFixed_Config_{test_id}_'):
                simulation_dir = dir
                break
        vcd_path = f"{hw_sim_path}/{simulation_dir}/BiasBlockFixedDUT.vcd"
        vcd_parser = VCDWaveformParser(vcd_path)
        simulation_results = vcd_parser.get_module_stats("Bias")
        simulation_latency = simulation_results['module_total_cycles']
        simulation_pipeline_depth = simulation_results['module_pipeline_depth_cycles']

        config_path = f"{test_folder_path}/config.json"
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        module = Bias(config["rows"],
                      config["cols"],
                      config["channels"],
                      config["channels"],
                      streams=config["streams"],
                      backend=BACKEND)

        modeling_latency = module.latency()
        modeling_pipeline_depth = module.pipeline_depth()

        assert modeling_latency == pytest.approx(simulation_latency, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling latency: {modeling_latency}, simulation latency: {simulation_latency}"
        assert modeling_pipeline_depth == pytest.approx(simulation_pipeline_depth, abs=ABS_TOL, rel=REL_TOL), f"TEST {test_id}: Modeling pipeline depth: {modeling_pipeline_depth}, simulation pipeline depth: {simulation_pipeline_depth}"
