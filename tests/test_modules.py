import glob
import unittest
import ddt
import json
import itertools

from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

ARCHS = [
        ( BACKEND.CHISEL, DIMENSIONALITY.TWO ),
        ( BACKEND.HLS, DIMENSIONALITY.TWO ),
        # ( BACKEND.HLS, DIMENSIONALITY.THREE ),
    ]

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
            self.assertGreaterEqual(module.rate_in(i), 0.0)
            self.assertLessEqual(module.rate_in(i), 1.0)
        # check rate out
        for i in range(module.ports_out):
            self.assertGreaterEqual(module.rate_out(i), 0.0)
            self.assertLessEqual(module.rate_out(i), 1.0)

    def run_test_latency(self, module):
        self.assertGreaterEqual(module.latency(), 1)

    def run_test_pipeline_depth(self, module):
        self.assertGreaterEqual(module.pipeline_depth(), 0)

    def run_test_resources(self, module):
        rsc = module.rsc()
        self.assertGreaterEqual(rsc["LUT"], 0.0)
        self.assertGreaterEqual(rsc["FF"], 0.0)
        self.assertGreaterEqual(rsc["DSP"], 0.0)
        self.assertGreaterEqual(rsc["BRAM"], 0.0)

# @ddt.ddt
# class TestForkModule(TestModuleTemplate,unittest.TestCase):

#     @ddt.data(*glob.glob("tests/configs/modules/fork/*.json"))
#     def test_module_configurations(self, config_path):
#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise module
#         module = Fork(config["rows"],config["cols"],config["channels"],
#                 config["kernel_size"],config["coarse"],backend=BACKEND)

#         # run tests
#         self.run_test_methods_exist(module)
#         self.run_test_dimensions(module)
#         self.run_test_rates(module)
#         self.run_test_resources(module)

@ddt.ddt
class TestAccumModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/accum/*.json")))
    def test_module_configurations(self, args):

        print(args)
        (backend, dimensionality), config_path = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = ModuleBase.build("accum", config,
                backend=backend, dimensionality=dimensionality)

        # run tests
        self.run_test_rates(module)
        self.run_test_latency(module)
        self.run_test_pipeline_depth(module)
        # self.run_test_resources(module)

        # # additional checks
        # self.assertGreater(module.filters, 0)
        # self.assertGreater(module.channle, 0)

# @ddt.ddt
# class TestConvModule(TestModuleTemplate,unittest.TestCase):

#     @ddt.data(*glob.glob("tests/configs/modules/conv/*.json"))
#     def test_module_configurations(self, config_path):

#         if BACKEND == "hls":

#             # open configuration
#             with open(config_path, "r") as f:
#                 config = json.load(f)

#             # initialise module
#             module = Conv(config["rows"],config["cols"],config["channels"],
#                     config["filters"],config["fine"],config["kernel_size"],
#                     config["group"],backend=BACKEND)

#             # run tests
#             self.run_test_methods_exist(module)
#             self.run_test_dimensions(module)
#             self.run_test_rates(module)
#             self.run_test_resources(module)

# @ddt.ddt
# class TestGlueModule(TestModuleTemplate,unittest.TestCase):

#     @ddt.data(*glob.glob("tests/configs/modules/glue/*.json"))
#     def test_module_configurations(self, config_path):
#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise module
#         module = Glue(config["rows"],config["cols"],config["channels"],
#                 config["filters"],config["coarse_in"],config["coarse_out"],backend=BACKEND)

#         # run tests
#         self.run_test_methods_exist(module)
#         self.run_test_dimensions(module)
#         self.run_test_rates(module)
#         self.run_test_resources(module)

# @ddt.ddt
# class TestSlidingWindowModule(TestModuleTemplate,unittest.TestCase):

#     @ddt.data(*glob.glob("tests/configs/modules/sliding_window/*.json"))
#     def test_module_configurations(self, config_path):
#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise module
#         module = SlidingWindow(config["rows"],config["cols"],config["channels"],
#                 config["kernel_size"],config["stride"],config["pad_top"],
#                 config["pad_right"],config["pad_bottom"],config["pad_left"],backend=BACKEND)

#         # run tests
#         self.run_test_methods_exist(module)
#         self.run_test_dimensions(module)
#         self.run_test_rates(module)
#         self.run_test_resources(module)

# @ddt.ddt
# class TestPoolModule(TestModuleTemplate,unittest.TestCase):

#     @ddt.data(*glob.glob("tests/configs/modules/pool/*.json"))
#     def test_module_configurations(self, config_path):
#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise module
#         module = Pool(config["rows"],config["cols"],config["channels"],
#                 config["kernel_size"],backend=BACKEND)

#         # run tests
#         self.run_test_methods_exist(module)
#         self.run_test_dimensions(module)
#         self.run_test_rates(module)
#         self.run_test_resources(module)

# @ddt.ddt
# class TestSqueezeModule(TestModuleTemplate,unittest.TestCase):

#     @ddt.data(*glob.glob("tests/configs/modules/squeeze/*.json"))
#     def test_module_configurations(self, config_path):
#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise module
#         module = Squeeze(config["rows"],config["cols"],config["channels"],
#                 config["coarse_in"],config["coarse_out"],backend=BACKEND)

#         # run tests
#         self.run_test_methods_exist(module)
#         self.run_test_dimensions(module)
#         self.run_test_rates(module)
#         self.run_test_resources(module)

# @ddt.ddt
# class TestReLUModule(TestModuleTemplate,unittest.TestCase):

#     @ddt.data(*glob.glob("tests/configs/modules/relu/*.json"))
#     def test_module_configurations(self, config_path):
#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise module
#         module = ReLU(config["rows"],config["cols"],config["channels"],backend=BACKEND)

#         # run tests
#         self.run_test_methods_exist(module)
#         self.run_test_dimensions(module)
#         self.run_test_rates(module)
#         self.run_test_resources(module)


