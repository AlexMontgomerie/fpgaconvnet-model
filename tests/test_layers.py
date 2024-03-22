import glob
import os
import unittest
import ddt
import json
import itertools
import numpy as np
import pytest

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fpgaconvnet.models.layers import  PoolingLayer, ConvolutionLayer, GlobalPoolingLayer, ReLULayer, InnerProductLayer, SqueezeLayer, SplitLayer, ConcatLayer, ReSizeLayer, EltWiseLayer

from pymongo import MongoClient
from pymongo.server_api import ServerApi

BACKEND = "chisel"
RESOURCES = ["LUT", "FF", "BRAM", "DSP"]

SERVER_DB="mongodb+srv://fpgaconvnet.hwnxpyo.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"

# absolute and relative tolerance
ABS_TOL = 200
REL_TOL = 0.05

def load_layer_configs_db():
    """
    Load layer configurations from the MongoDB database
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
    collection = database["test-configurations-chisel"]

    # find all configurations for the given name
    configs = collection.find({"hierarchy": "layer"})

    # return the configurations
    return list(configs)


def initialise_layers(configs):
    """
    Initialise the layers from the configurations
    """

    # list of layers
    layers = []

    # initialise the layers
    for config in configs:

        if config["name"] == "average_pooling": config["name"] = "global_pool"

        # FIXME: get backend and dimensionality from the config
        config["backend"] = "CHISEL"
        config["dimensionality"] = 2

        # copy *_in to * for compatibility
        if "rows" not in config: config["rows"] = config.get("rows_in", 1)
        if "cols" not in config: config["cols"] = config.get("cols_in", 1)
        if "channels" not in config: config["channels"] = config.get("channels_in", 1)

        # build the layer
        match config["name"]:
            case "pooling":
                layer = PoolingLayer(
                    config["rows"],
                    config["cols"],
                    config["channels"],
                    coarse=config["coarse"],
                    kernel_rows=config["kernel_size"][0],
                    kernel_cols=config["kernel_size"][1],
                    stride_rows=config["stride"][0],
                    stride_cols=config["stride"][1],
                    pad_top=config["pad_top"],
                    pad_left=config["pad_left"],
                    pad_bottom=config["pad_bottom"],
                    pad_right=config["pad_right"],
                )
            case "convolution":
                layer = ConvolutionLayer(
                    config["filters"],
                    config["rows"],
                    config["cols"],
                    config["channels"],
                    coarse_in=config["coarse_in"],
                    coarse_out=config["coarse_out"],
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
                    has_bias=True
                )
            case "relu":
                layer = ReLULayer(
                    config["rows"],
                    config["cols"],
                    config["channels"],
                    coarse = config["coarse"]
                )
            case "inner_product":
                layer = InnerProductLayer(
                    config["filters"],
                    config["rows"],
                    config["cols"],
                    config["channels"],
                    config["coarse_in"],
                    config["coarse_out"],
                    has_bias=True
                )
            case "squeeze":
                layer = SqueezeLayer(
                    config["rows"],
                    config["cols"],
                    config["channels"],
                    config["coarse_in"],
                    config["coarse_out"],
                )
            case "split":
                layer = SplitLayer(
                    config["rows"],
                    config["cols"],
                    config["channels"],
                    coarse=config["coarse"],
                    ports_out=config["ports_out"]
                )
            case "concat":
                config["channels"] = config.get("channels_in_array", [1])
                layer = ConcatLayer(
                    config["rows"],
                    config["cols"],
                    config["channels"],
                    ports_in=config["ports_in"],
                    coarse=config["coarse"]
                )
            case "resize":
                layer = ReSizeLayer(
                    config["rows"],
                    config["cols"],
                    config["channels"],
                    config["scales"],
                    coarse=config["coarse"],
                )
            case "eltwise":
                layer = EltWiseLayer(
                    config["rows"],
                    config["cols"],
                    config["channels"],
                    coarse=config["coarse"],
                    ports_in=config["ports_in"],
                )
            case "global_pool":
                layer = GlobalPoolingLayer(
                    config["rows"],
                    config["cols"],
                    config["channels"],
                    coarse=config["coarse"]
                )
            case _:
                raise ValueError(f"Unknown layer type: {config['name']}")
        # create a name for the layer
        name = f"layer:{BACKEND}:2D:{config['name']} {config['_id']}"

        # append the layer
        layers.append((name, layer, config))

    # return the layers
    return layers

CONFIGS=load_layer_configs_db()
LAYERS=initialise_layers(CONFIGS)

@ddt.ddt
class TestModule(unittest.TestCase):

    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_rates(self, layer, config: dict):
        pass
        # # check rate in
        # for i in range(layer.ports_in):
        #     self.assertGreaterEqual(layer.get_rate_in(i), 0.0)
        #     self.assertLessEqual(layer.get_rate_in(i), 1.0)
        # # check rate out
        # for i in range(layer.ports_out):
        #     self.assertGreaterEqual(layer.get_rate_out(i), 0.0)
        #     self.assertLessEqual(layer.get_rate_out(i), 1.0)


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_dimensions(self, layer, config: dict):
        # check input dimensions
        self.assertTrue(layer.rows_in() > 0)
        self.assertTrue(layer.cols_in() > 0)
        self.assertTrue(layer.channels_in() > 0)
        # if layer.dimensionality == DIMENSIONALITY.THREE:
        #     self.assertTrue(layer.depth_in() > 0)
        # check output dimensions
        self.assertTrue(layer.rows_out() > 0)
        self.assertTrue(layer.cols_out() > 0)
        self.assertTrue(layer.channels_out() > 0)
        # if layer.dimensionality == DIMENSIONALITY.THREE:
        #     self.assertTrue(layer.depth_out() > 0)


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_rates(self, layer, config: dict):

        # check rate in
        assert layer.rate_in() >= 0.0, f"Rate in is negative"
        assert layer.rate_in() <= 1.0, f"Rate in is greater than 1.0"

        # check rate out
        assert layer.rate_out() >= 0.0, f"Rate out is negative"
        assert layer.rate_out() <= 1.0, f"Rate out is greater than 1.0"


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_workload(self, layer, config: dict):
        # check workload in
        self.assertTrue(layer.workload_in() >= 0.0)
        # check workload out
        self.assertTrue(layer.workload_out() >= 0.0)


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_streams(self, layer, config: dict):
        # check streams in
        self.assertTrue(layer.streams_in() >= 1)
        # check streams out
        self.assertTrue(layer.streams_out() >= 1)


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_size(self, layer, config: dict):
        # check size in
        self.assertTrue(layer.size_in() >= 1)
        # check size out
        self.assertTrue(layer.size_out() >= 1)


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_latency(self, layer, config: dict):
        # check latency
        self.assertTrue(layer.latency() >= 0.0)


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_pipeline_depth(self, layer, config: dict):
        # check pipeline depth
        assert layer.pipeline_depth() >= 0.0, f"Pipeline depth is negative"


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_wait_depth(self, layer, config: dict):
        # check wait depth
        #self.assertTrue(layer.wait_depth() >= 0.0)
        pass


    @ddt.unpack
    @ddt.named_data(*[  (f"{name} ({rsc_type})", rsc_type, layer, config) \
            for rsc_type, (name, layer, config) in itertools.product(RESOURCES, LAYERS) ])
    def test_resources(self, rsc_type, layer, config):

        # check the resources
        actual_rsc= config["resource"][rsc_type]
        modelled_rsc= layer.resource()[rsc_type]

        assert modelled_rsc >= 0
        assert modelled_rsc == pytest.approx(actual_rsc, abs=ABS_TOL, rel=REL_TOL), \
            f"Resource {rsc_type} does not match. Modelled: {modelled_rsc}, Actual: {actual_rsc}"


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_cycles(self, layer, config):

        # get the cycles
        cycles = config["cycles"]

        # get the modelled cycles
        model_cycles = layer.latency()

        # check the cycles
        if cycles > 0:
            assert model_cycles == pytest.approx(cycles, abs=ABS_TOL, rel=REL_TOL), \
                f"Modelled cycles do not match. Expected: {cycles}, Actual: {model_cycles}"


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_updating_properties(self, layer, config: dict):

        # updating coarse in
        coarse_in = max(layer.get_coarse_in_feasible())
        layer.coarse_in = coarse_in
        assert layer.coarse_in == coarse_in, f"Coarse in does not match. Expected: {coarse_in}, Actual: {layer.coarse_in}"

        # updating coarse out
        coarse_out = max(layer.get_coarse_out_feasible())
        layer.coarse_out = coarse_out
        assert layer.coarse_out == coarse_out, f"Coarse out does not match. Expected: {coarse_out}, Actual: {layer.coarse_out}"


