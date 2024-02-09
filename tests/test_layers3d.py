import glob
import unittest
import ddt
import json
from fpgaconvnet.models.layers import *

class TestLayer3DTemplate():

    def run_test_dimensions(self, layer):
        # check input dimensions
        self.assertTrue(layer.rows_in() > 0)
        self.assertTrue(layer.cols_in() > 0)
        self.assertTrue(layer.depth_in() > 0)
        self.assertTrue(layer.channels_in() > 0)
        # check output dimensions
        self.assertTrue(layer.rows_out() > 0)
        self.assertTrue(layer.cols_out() > 0)
        self.assertTrue(layer.depth_out() > 0)
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


@ddt.ddt
class TestPoolingLayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/pooling3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = PoolingLayer3D(
            config["rows"],
            config["cols"],
            config["depth"],
            config["channels"],
            coarse=config["coarse"],
            kernel_rows=config["kernel_rows"],
            kernel_cols=config["kernel_cols"],
            kernel_depth=config["kernel_depth"],
            stride_rows=config["stride_rows"],
            stride_cols=config["stride_cols"],
            stride_depth=config["stride_depth"],
            pad_top=config["pad_top"],
            pad_right=config["pad_right"],
            pad_front=config["pad_front"],
            pad_bottom=config["pad_bottom"],
            pad_left=config["pad_left"],
            pad_back=config["pad_back"],
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
class TestConvolutionLayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/convolution3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = ConvolutionLayer3D(
            config["filters"],
            config["rows"],
            config["cols"],
            config["depth"],
            config["channels"],
            coarse_in=config["coarse_in"],
            coarse_out=config["coarse_out"],
            kernel_rows=config["kernel_rows"],
            kernel_cols=config["kernel_cols"],
            kernel_depth=config["kernel_depth"],
            stride_rows=config["stride_rows"],
            stride_cols=config["stride_cols"],
            stride_depth=config["stride_depth"],
            groups=config["groups"],
            pad_top=config["pad_top"],
            pad_right=config["pad_right"],
            pad_front=config["pad_front"],
            pad_bottom=config["pad_bottom"],
            pad_left=config["pad_left"],
            pad_back=config["pad_back"],
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
class TestActivationLayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/activation3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = ActivationLayer3D(
            config["rows"],
            config["cols"],
            config["depth"],
            config["channels"],
            config["activation_type"],
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
class TestReLULayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/relu3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = ReLULayer3D(
            config["rows"],
            config["cols"],
            config["depth"],
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
class TestInnerProductLayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/inner_product3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = InnerProductLayer3D(
            config["filters"],
            config["rows"],
            config["cols"],
            config["depth"],
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
class TestSqueezeLayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/squeeze3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = SqueezeLayer3D(
            config["rows"],
            config["cols"],
            config["depth"],
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
class TestGlobalPoolingLayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/avgpool3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = GlobalPoolingLayer3D(
            config["rows"],
            config["cols"],
            config["depth"],
            config["channels"],
            config["coarse"],
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
class TestEltWiseLayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/eltwise3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = EltWiseLayer3D(
            config["rows"],
            config["cols"],
            config["depth"],
            config["channels"],
            config["ports_in"],
            config["coarse"],
            config["eltwise_type"],
            config["broadcast"],
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
        # self.run_test_updating_properties(layer)
        self.run_test_resources(layer)