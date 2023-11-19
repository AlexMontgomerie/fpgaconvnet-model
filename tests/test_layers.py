import unittest
import ddt
import json
import glob

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

# get the paths for all the layer configs
RELU_CONF_PATH=list(glob.glob("tests/configs/layers/relu/*"))
CONVOLUTION_CONF_PATH=list(glob.glob("tests/configs/layers/convolution/*"))
POOLING_CONF_PATH=list(glob.glob("tests/configs/layers/pooling/*"))
INNERPRODUCT_CONF_PATH=list(glob.glob("tests/configs/layers/inner_product/*"))
SQUEEZE_CONF_PATH=list(glob.glob("tests/configs/layers/squeeze/*"))
HARDSWISH_CONF_PATH=list(glob.glob("tests/configs/layers/hardswish/*"))

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
        # self.assertEqual(set(list(rsc.keys())), set(["URAM", "BRAM","DSP","LUT","FF"]))
        self.assertTrue(rsc["LUT"] >= 0)
        self.assertTrue(rsc["FF"] >= 0)
        self.assertTrue(rsc["DSP"] >= 0)
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


# @ddt.ddt
# class TestPoolingLayer(TestLayerTemplate,unittest.TestCase):

#     @ddt.data(*POOLING_CONF_PATH)
#     def test_layer_configurations(self, config_path):

#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # get the kernel rows and cols
#         config["kernel_rows"] = config["kernel_size"][0]
#         config["kernel_cols"] = config["kernel_size"][1]

#         # get the stride rows and cols
#         config["stride_rows"] = config["stride"][0]
#         config["stride_cols"] = config["stride"][1]

#         # initialise layer
#         layer = Layer.build_from_dict("PoolingLayer", config)
#         # layer = PoolingLayer(
#         #     rows=config["rows"],
#         #     cols=config["cols"],
#         #     channels=config["channels"],
#         #     coarse=config["coarse"],
#         #     kernel_rows=config["kernel_size"][0],
#         #     kernel_cols=config["kernel_size"][1],
#         #     stride_rows=config["stride"][0],
#         #     stride_cols=config["stride"][1],
#         #     pad_top=config["pad"],
#         #     pad_left=config["pad"],
#         #     pad_bottom=config["pad"],
#         #     pad_right=config["pad"],
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
class TestConvolutionLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(*CONVOLUTION_CONF_PATH)
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # get the kernel rows and cols
        config["kernel_rows"] = config["kernel_size"][0]
        config["kernel_cols"] = config["kernel_size"][1]

        # get the stride rows and cols
        config["stride_rows"] = config["stride"][0]
        config["stride_cols"] = config["stride"][1]

        # initialise layer
        layer = LayerBase.build("convolution", config, BACKEND.CHISEL, DIMENSIONALITY.TWO)
        # layer = ConvolutionLayer(
        #     filters=config["filters"],
        #     rows=config["rows"],
        #     cols=config["cols"],
        #     channels=config["channels"],
        #     coarse_in=config["coarse_in"],
        #     coarse_out=config["coarse_out"],
        #     kernel_rows=config["kernel_size"][0],
        #     kernel_cols=config["kernel_size"][1],
        #     stride_rows=config["stride"][0],
        #     stride_cols=config["stride"][1],
        #     groups=config["groups"],
        #     pad_top=config["pad"],
        #     pad_left=config["pad"],
        #     pad_bottom=config["pad"],
        #     pad_right=config["pad"],
        #     fine=config["fine"],
        # )

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
        # self.run_test_resources(layer)

# @ddt.ddt
# class TestReLULayer(TestLayerTemplate,unittest.TestCase):

#     @ddt.data(*RELU_CONF_PATH)
#     def test_layer_configurations(self, config_path):

#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise layer
#         layer = Layer.build_from_dict("ReLULayer", config)
#         # layer = ReLULayer(
#         #     rows=config["rows"],
#         #     cols=config["cols"],
#         #     channels=config["channels"],
#         #     coarse=config["coarse"]
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

# @ddt.ddt
# class TestSqueezeLayer(TestLayerTemplate,unittest.TestCase):

#     @ddt.data(*SQUEEZE_CONF_PATH)
#     def test_layer_configurations(self, config_path):

#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise layer
#         layer = Layer.build_from_dict("SqueezeLayer", config)
#         # layer = SqueezeLayer(
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

# @ddt.ddt
# class TestHardswishLayer(TestLayerTemplate,unittest.TestCase):

#     @ddt.data(*HARDSWISH_CONF_PATH)
#     def test_layer_configurations(self, config_path):

#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise layer
#         layer = Layer.build_from_dict("HardswishLayer", config)
#         # layer = HardswishLayer(
#         #     rows=config["rows"],
#         #     cols=config["cols"],
#         #     channels=config["channels"],
#         #     coarse=config["coarse"],
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


# # @ddt.ddt
# # class TestSplitLayer(TestLayerTemplate,unittest.TestCase):

# #     @ddt.data(
# #         "tests/configs/layers/split/config_0.json",
# #     )
# #     def test_layer_configurations(self, config_path):

# #         # open configuration
# #         with open(config_path, "r") as f:
# #             config = json.load(f)

# #         # initialise layer
# #         layer = SplitLayer(
# #             config["rows"],
# #             config["cols"],
# #             config["channels"],
# #             config["coarse"],
# #             ports_out=config["ports_out"]
# #         )

# #         # run tests
# #         self.run_test_dimensions(layer)
# #         self.run_test_rates(layer)
# #         self.run_test_workload(layer)
# #         self.run_test_size(layer)
# #         self.run_test_streams(layer)
# #         self.run_test_latency(layer)
# #         self.run_test_pipeline_depth(layer)
# #         self.run_test_wait_depth(layer)
# #         self.run_test_resources(layer)

