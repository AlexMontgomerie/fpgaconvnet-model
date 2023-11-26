import glob
import unittest
import ddt
import json
import itertools
import numpy as np

from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.models.modules.resources import eval_resource_model
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.exceptions import ModuleNotImplementedError

ARCHS = [
        ( BACKEND.CHISEL, DIMENSIONALITY.TWO ),
        ( BACKEND.CHISEL, DIMENSIONALITY.THREE ),
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
        module = ModuleBase.build(name, params, backend=backend, dimensionality=dimensionality)
        self.assertTrue(isinstance(module, type(module)))

@ddt.ddt
class TestForkModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/fork/*.json")))
    def test_module_configurations(self, args):

        (backend, dimensionality), config_path = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        config["fine"] = int(np.prod(config["kernel_size"]))

        if isinstance(config["kernel_size"], int):
            config["kernel_size"] = [config["kernel_size"]]*dimensionality.value

        # initialise module
        module = ModuleBase.build("fork", config,
                backend=backend, dimensionality=dimensionality)

        # run tests
        self.run_test_rates(module)
        self.run_test_latency(module)
        self.run_test_pipeline_depth(module)
        self.run_test_config_gen(module)
        self.run_test_resources(module)

        # # additional checks
        # self.assertGreater(module.filters, 0)
        # self.assertGreater(module.channle, 0)

@ddt.ddt
class TestAccumModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/accum/*.json")))
    def test_module_configurations(self, args):

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
        self.run_test_config_gen(module)
        self.run_test_resources(module)

@ddt.ddt
class TestConvModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/conv/*.json")))
    def test_module_configurations(self, args):

        (backend, dimensionality), config_path = args

        if backend == BACKEND.HLS:

            # open configuration
            with open(config_path, "r") as f:
                config = json.load(f)

            if isinstance(config["kernel_size"], int):
                config["kernel_size"] = [config["kernel_size"]]*dimensionality.value

            # initialise module
            module = ModuleBase.build("conv", config,
                    backend=backend, dimensionality=dimensionality)

            # run tests
            self.run_test_rates(module)
            self.run_test_latency(module)
            self.run_test_pipeline_depth(module)
            self.run_test_config_gen(module)
            self.run_test_resources(module)

@ddt.ddt
class TestGlueModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/glue/*.json")))
    def test_module_configurations(self, args):

        (backend, dimensionality), config_path = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        config["coarse_group"] = 1
        config["coarse"] = config["coarse_in"]

        # initialise module
        module = ModuleBase.build("glue", config,
                backend=backend, dimensionality=dimensionality)

        # run tests
        self.run_test_rates(module)
        self.run_test_latency(module)
        self.run_test_pipeline_depth(module)
        self.run_test_config_gen(module)
        self.run_test_resources(module)

@ddt.ddt
class TestSlidingWindowModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/sliding_window/*.json")))
    def test_module_configurations(self, args):

        (backend, dimensionality), config_path = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        if isinstance(config["kernel_size"], int):
            config["kernel_size"] = [config["kernel_size"]]*dimensionality.value
        if isinstance(config["stride"], int):
            config["stride"] = [config["stride"]]*dimensionality.value
        if "pad" in config:
            if isinstance(config["pad"], int):
                config["pad"] = [config["pad"]]*dimensionality.value*2
        else:
            config["pad"] = [0]*dimensionality.value*2

        try:
            # initialise module
            module = ModuleBase.build("sliding_window", config,
                    backend=backend, dimensionality=dimensionality)

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

    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/pool/*.json")))
    def test_module_configurations(self, args):

        (backend, dimensionality), config_path = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # set the pool type
        config["pool_type"] = "max"
        if isinstance(config["kernel_size"], int):
            config["kernel_size"] = [config["kernel_size"]]*dimensionality.value

        try:
            # initialise module
            module = ModuleBase.build("pool", config,
                    backend=backend, dimensionality=dimensionality)

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

    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/squeeze/*.json")))
    def test_module_configurations(self, args):

        (backend, dimensionality), config_path = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = ModuleBase.build("squeeze", config,
                backend=backend, dimensionality=dimensionality)

        # run tests
        self.run_test_rates(module)
        self.run_test_latency(module)
        self.run_test_pipeline_depth(module)
        self.run_test_config_gen(module)
        self.run_test_resources(module)

@ddt.ddt
class TestReLUModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*itertools.product(ARCHS, glob.glob("tests/configs/modules/relu/*.json")))
    def test_module_configurations(self, args):

        (backend, dimensionality), config_path = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = ModuleBase.build("relu", config,
                backend=backend, dimensionality=dimensionality)

        # run tests
        self.run_test_rates(module)
        self.run_test_latency(module)
        self.run_test_pipeline_depth(module)
        self.run_test_config_gen(module)
        self.run_test_resources(module)


