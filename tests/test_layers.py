import unittest
import ddt
import json
import glob
import pytest
import itertools

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.exceptions import LayerNotImplementedError, AmbiguousLayerError, ModuleNotImplementedError

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
ARCHITECTURES = [
    (BACKEND.CHISEL, DIMENSIONALITY.TWO),
    (BACKEND.CHISEL, DIMENSIONALITY.THREE),
    (BACKEND.HLS, DIMENSIONALITY.TWO),
    # (BACKEND.HLS, DIMENSIONALITY.THREE),
]

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

    def run_test_updating_properties(self, layer):
        # updating coarse in
        coarse_in = max(layer.get_coarse_in_feasible())
        layer.coarse_in = coarse_in
        self.assertEqual(layer.coarse_in, coarse_in)
        # updating coarse out
        coarse_out = max(layer.get_coarse_out_feasible())
        layer.coarse_out = coarse_out
        self.assertEqual(layer.coarse_out, coarse_out)


@ddt.ddt
# @pytest.mark.skip(reason="Not implemented yet")
class TestPoolingLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(*list(itertools.product(POOLING_CONF_PATH, ARCHITECTURES)))
    def test_layer_configurations(self, args):

        # extract the arguments
        config_path, (backend, dimensionality) = args

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
        if dimensionality == DIMENSIONALITY.THREE:
            config["kernel_depth"] = config["kernel_size"][1]
            config["stride_depth"] = config["stride"][1]
            config["depth"] = config["cols"]

        try:
            # initialise layer
            layer = LayerBase.build("pooling", config, backend, dimensionality)

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

    @ddt.data(*list(itertools.product(CONCAT_CONF_PATH, ARCHITECTURES)))
    def test_layer_configurations(self, args):

        # extract the arguments
        config_path, (backend, dimensionality) = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # add dimensionality information
        if dimensionality == DIMENSIONALITY.THREE:
            config["depth"] = config["cols"]

        try:
            # initialise layer
            layer = LayerBase.build("concat", config, backend, dimensionality)

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

    @ddt.data(*list(itertools.product(CONVOLUTION_CONF_PATH, ARCHITECTURES)))
    def test_layer_configurations(self, args):

        # extract the arguments
        config_path, (backend, dimensionality) = args

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
        if dimensionality == DIMENSIONALITY.THREE:
            config["kernel_depth"] = config["kernel_size"][1]
            config["stride_depth"] = config["stride"][1]
            config["depth"] = config["cols"]

        try:
            # initialise layer
            layer = LayerBase.build("convolution", config, backend, dimensionality)

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

    @ddt.data(*list(itertools.product(RELU_CONF_PATH, ARCHITECTURES)))
    def test_layer_configurations(self, args):

        # extract the arguments
        config_path, (backend, dimensionality) = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # add dimensionality information
        if dimensionality == DIMENSIONALITY.THREE:
            config["depth"] = config["cols"]

        try:
            # initialise layer
            layer = LayerBase.build("relu", config, backend, dimensionality)

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

    @ddt.data(*list(itertools.product(SQUEEZE_CONF_PATH, ARCHITECTURES)))
    def test_layer_configurations(self, args):

        # extract the arguments
        config_path, (backend, dimensionality) = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # add dimensionality information
        if dimensionality == DIMENSIONALITY.THREE:
            config["depth"] = config["cols"]

        try:
            # initialise layer
            layer = LayerBase.build("squeeze", config, backend, dimensionality)

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

    @ddt.data(*list(itertools.product(HARDSWISH_CONF_PATH, ARCHITECTURES)))
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

    @ddt.data(*list(itertools.product(SPLIT_CONF_PATH, ARCHITECTURES)))
    def test_layer_configurations(self, args):

        # extract the arguments
        config_path, (backend, dimensionality) = args

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # add dimensionality information
        if dimensionality == DIMENSIONALITY.THREE:
            config["depth"] = config["cols"]

        # initialise layer
        try:
            layer = LayerBase.build("split", config, backend, dimensionality)

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


