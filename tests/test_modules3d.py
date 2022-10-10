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