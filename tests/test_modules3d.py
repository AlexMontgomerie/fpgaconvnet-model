import glob
import unittest
import ddt
import json
from fpgaconvnet.models.modules import *


class TestModule3DTemplate():

    def run_test_methods_exist(self, module):
        self.assertTrue(hasattr(module, "rows_in"))
        self.assertTrue(hasattr(module, "cols_in"))
        self.assertTrue(hasattr(module, "depth_in"))
        self.assertTrue(hasattr(module, "channels_in"))
        self.assertTrue(hasattr(module, "rows_out"))
        self.assertTrue(hasattr(module, "cols_out"))
        self.assertTrue(hasattr(module, "depth_out"))
        self.assertTrue(hasattr(module, "channels_out"))

    def run_test_dimensions(self, module):
        # check input dimensions
        self.assertGreater(module.rows_in(), 0)
        self.assertGreater(module.cols_in(), 0)
        self.assertGreater(module.depth_in(), 0)
        self.assertGreater(module.channels_in(), 0)
        # check output dimensions
        self.assertGreater(module.rows_out(), 0)
        self.assertGreater(module.cols_out(), 0)
        self.assertGreater(module.depth_out(), 0)
        self.assertGreater(module.channels_out(), 0)

    def run_test_rates(self, module):
        # check rate in
        self.assertGreaterEqual(module.rate_in(), 0.0)
        self.assertLessEqual(module.rate_in(),1.0)
        # check rate out
        self.assertGreaterEqual(module.rate_out(), 0.0)
        self.assertLessEqual(module.rate_out(), 1.0)

    def run_test_resources(self, module):
        rsc = module.rsc()
        self.assertGreaterEqual(rsc["LUT"], 0.0)
        self.assertGreaterEqual(rsc["FF"], 0.0)
        self.assertGreaterEqual(rsc["DSP"], 0.0)
        self.assertGreaterEqual(rsc["BRAM"], 0.0)

@ddt.ddt
class TestReLU3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/relu3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = ReLU3D(config["rows"],config["cols"],config["depth"],config["channels"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestActivation3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/activation3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Activation3D(config["rows"],config["cols"],config["depth"],config["channels"],config["type"].lower())

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestAccum3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/accum3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Accum3D(config["rows"],config["cols"],config["depth"],config["channels"],config["filters"],config["groups"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestBias3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/bias3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Bias3D(config["rows"],config["cols"],config["depth"],config["channels"],config["filters"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestConv3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/conv3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Conv3D(config["rows"],config["cols"],config["depth"],config["channels"],
                config["filters"],config["fine"],config["kernel_rows"],config["kernel_cols"],config["kernel_depth"],config["group"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestGlobalPool3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/avgpool3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = GlobalPool3D(config["rows"],config["cols"],config["depth"],config["channels"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestFork3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/fork3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Fork3D(config["rows"],config["cols"],config["depth"],config["channels"],config["kernel_rows"],config["kernel_cols"],config["kernel_depth"],config["coarse"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestGlue3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/glue3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Glue3D(config["rows"],config["cols"],config["depth"],config["channels"],config["filters"],config["coarse_in"],config["coarse_out"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestPool3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/pool3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Pool3D(config["rows"],config["cols"],config["depth"],config["channels"],config["kernel_rows"],config["kernel_cols"],config["kernel_depth"],config["pool_type"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestSqueeze3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/squeeze3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Squeeze3D(config["rows"],config["cols"],config["depth"],config["channels"],config["coarse_in"],config["coarse_out"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)


@ddt.ddt
class TestVectorDot3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/vector_dot3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = VectorDot3D(config["rows"],config["cols"],config["depth"],config["channels"],config["filters"],config["fine"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestEltWise3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/eltwise3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = EltWise3D(config["rows"],config["cols"],config["depth"],config["ports_in"],config["eltwise_type"],config["broadcast"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestSlidingWindow3DModule(TestModule3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/sliding_window3d/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = SlidingWindow3D(config["rows"],config["cols"],config["depth"],config["channels"],config["kernel_rows"],config["kernel_cols"],config["kernel_depth"],config["stride_rows"],config["stride_cols"],config["stride_depth"],config["pad_top"],config["pad_right"],config["pad_front"],config["pad_bottom"],config["pad_left"],config["pad_back"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)
