import glob
import unittest
import ddt
import json
from fpgaconvnet.models.modules import *
import glob
import os
from fpgaconvnet.tools.waveform_parser import VCDWaveformParser
import pytest

from pymongo import MongoClient
from pymongo.server_api import ServerApi

SERVER_DB="mongodb+srv://fpgaconvnet.hwnxpyo.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"

# Define the path to the hardware backend directory (fpgaconvnet-chisel)
HW_BACKEND_PATH = "../fpgaconvnet-chisel"
ABS_TOL = 200
REL_TOL = 0.05
BACKEND = "chisel"

def load_module_configs_db(name):
    """
    Load module configurations from the MongoDB database
    """

    # database .pem path
    db_pem = os.path.join(os.path.dirname(__file__),
            "fpgaconvnet-mongodb.pem")

    # create MongoDB client
    client = MongoClient(SERVER_DB, tls=True,
        tlsCertificateKeyFile=db_pem,
        server_api=ServerApi('1'))

    # open the database
    database = client["fpgaconvnet"]

    # open the collection
    collection = database["chisel-latest"]

    # find all configurations for the given name
    configs = collection.find({"name": name})

    # # close the client
    # client.close()

    # return the configurations
    return list(configs)



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
        self.assertGreaterEqual(module.rate_in(), 0.0)
        self.assertLessEqual(module.rate_in(),1.0)
        # check rate out
        self.assertGreaterEqual(module.rate_out(), 0.0)
        self.assertLessEqual(module.rate_out(), 1.0)

    def run_test_resources(self, module, resource={}):

        # run the resource model
        rsc = module.rsc()

        # make sure the resources are greater than zero
        self.assertGreaterEqual(rsc["LUT"], 0.0)
        self.assertGreaterEqual(rsc["FF"], 0.0)
        self.assertGreaterEqual(rsc["DSP"], 0.0)
        self.assertGreaterEqual(rsc["BRAM"], 0.0)

        # check the resources
        for rsc_type in rsc:
            expected_rsc = resource.get(rsc_type, rsc[rsc_type])
            assert rsc[rsc_type] == pytest.approx(expected_rsc, abs=ABS_TOL, rel=REL_TOL), \
                f"Resource {rsc_type} does not match. Expected: {expected_rsc}, Actual: {rsc[rsc_type]}"

    def run_test_cycles(self, module, cycles=0):

        # get the modelled cycles
        model_cycles = module.latency()

        # check the cycles
        if cycles > 0:
            assert model_cycles == pytest.approx(cycles, abs=ABS_TOL, rel=REL_TOL), \
                f"Modelled cycles do not match. Expected: {cycles}, Actual: {model_cycles}"

    def run_hw_simulation(self, layer, index):
        # run hardware simulation
        os.system(f"python {HW_BACKEND_PATH}/scripts/data/generate_module_block_data.py -l {layer} -n {index} -p {HW_BACKEND_PATH}")
        # sbt "testOnly fpgaconvnet.layers.${name}_test.ConfigTest"
        os.system(f"cd {HW_BACKEND_PATH} && sbt -Dconfig_idx={index} 'testOnly fpgaconvnet.layers.{layer}_block_test.ConfigTest' && cd -")

@ddt.ddt
class TestForkModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/fork/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Fork(config["rows"],config["cols"],config["channels"],
                config["kernel_size"],config["coarse"],backend=BACKEND)

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestAccumModule(TestModuleTemplate,unittest.TestCase):

    # @ddt.data(*glob.glob("tests/configs/modules/accum/*.json"))
    @ddt.data(*load_module_configs_db("accum"))
    def test_module_configurations(self, config):

        # # open configuration
        # with open(config_path, "r") as f:
        #     config = json.load(f)

        config["rows"] = config.get("rows", 1)
        config["cols"] = config.get("cols", 1)
        config["groups"] = config.get("groups", 1)

        # initialise module
        module = Accum(config["rows"],config["cols"],config["channels"],
                config["filters"],config["groups"],backend=BACKEND)

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module, config["resource"])
        self.run_test_cycles(module, config["cycles"])

@ddt.ddt
class TestConvModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/conv/*.json"))
    def test_module_configurations(self, config_path):

        if BACKEND == "hls":

            # open configuration
            with open(config_path, "r") as f:
                config = json.load(f)

            # initialise module
            module = Conv(config["rows"],config["cols"],config["channels"],
                    config["filters"],config["fine"],config["kernel_size"],
                    config["group"],backend=BACKEND)

            # run tests
            self.run_test_methods_exist(module)
            self.run_test_dimensions(module)
            self.run_test_rates(module)
            self.run_test_resources(module)

@ddt.ddt
class TestGlueModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/glue/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Glue(config["rows"],config["cols"],config["channels"],
                config["filters"],config["coarse_in"],config["coarse_out"],config["coarse_group"],backend=BACKEND)

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestSlidingWindowModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/sliding_window/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = SlidingWindow(config["rows"],config["cols"],config["channels"],
                config["kernel_size"],config["stride"],config["pad_top"],
                config["pad_right"],config["pad_bottom"],config["pad_left"],backend=BACKEND)

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestPoolModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/pool/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Pool(config["rows"],config["cols"],config["channels"],
                config["kernel_size"],backend=BACKEND)

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestSqueezeModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/squeeze/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Squeeze(config["rows"],config["cols"],config["channels"],
                config["coarse_in"],config["coarse_out"],backend=BACKEND)

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestReLUModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/relu/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = ReLU(config["rows"],config["cols"],config["channels"],backend=BACKEND)

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)


@unittest.skip("Currently disabled in CI.")
@ddt.ddt
@pytest.mark.skip(reason="Not implemented yet")
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

@unittest.skip("Currently disabled in CI.")
@ddt.ddt
@pytest.mark.skip(reason="Not implemented yet")
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

@unittest.skip("Currently disabled in CI.")
@ddt.ddt
@pytest.mark.skip(reason="Not implemented yet")
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

@unittest.skip("Currently disabled in CI.")
@ddt.ddt
@pytest.mark.skip(reason="Not implemented yet")
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

@unittest.skip("Currently disabled in CI.")
@ddt.ddt
@pytest.mark.skip(reason="Not implemented yet")
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

@unittest.skip("Currently disabled in CI.")
@ddt.ddt
@pytest.mark.skip(reason="Not implemented yet")
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
