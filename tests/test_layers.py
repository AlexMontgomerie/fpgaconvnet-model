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

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY

from pymongo import MongoClient
from pymongo.server_api import ServerApi

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

        # get the architecture
        arch = Architecture(BACKEND[config["backend"]],
                            DIMENSIONALITY(config["dimensionality"]))

        # build the layer
        layer = LayerBase.build(config["name"], config, arch.backend, arch.dimensionality)

        # create a name for the layer
        name = f"layer:{arch.backend.name}:{arch.dimensionality.name}:{config['name']} {config['_id']}"

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
    def test_rates(self, layer: LayerBase, config: dict):
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
    def test_dimensions(self, layer: LayerBase, config: dict):
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


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_rates(self, layer: LayerBase, config: dict):

        # check rate in
        assert layer.rate_in() >= 0.0, f"Rate in is negative"
        assert layer.rate_in() <= 1.0, f"Rate in is greater than 1.0"

        # check rate out
        assert layer.rate_out() >= 0.0, f"Rate out is negative"
        assert layer.rate_out() <= 1.0, f"Rate out is greater than 1.0"


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_workload(self, layer: LayerBase, config: dict):
        # check workload in
        self.assertTrue(layer.workload_in() >= 0.0)
        # check workload out
        self.assertTrue(layer.workload_out() >= 0.0)


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_streams(self, layer: LayerBase, config: dict):
        # check streams in
        self.assertTrue(layer.streams_in() >= 1)
        # check streams out
        self.assertTrue(layer.streams_out() >= 1)


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_size(self, layer: LayerBase, config: dict):
        # check size in
        self.assertTrue(layer.size_in() >= 1)
        # check size out
        self.assertTrue(layer.size_out() >= 1)


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_latency(self, layer: LayerBase, config: dict):
        # check latency
        self.assertTrue(layer.latency() >= 0.0)


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_pipeline_depth(self, layer: LayerBase, config: dict):
        # check pipeline depth
        assert layer.pipeline_depth() >= 0.0, f"Pipeline depth is negative"


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_wait_depth(self, layer: LayerBase, config: dict):
        # check wait depth
        #self.assertTrue(layer.wait_depth() >= 0.0)
        pass


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_resources(self, layer, config):

        # get the resource model
        resource_actual = config["resource"]
        resource_modelled = layer.resource()

        # check the resources
        for rsc_type in ["LUT", "FF", "BRAM"]:
            assert resource_modelled[rsc_type] >= 0, f"Resource {rsc_type} is negative"
            assert resource_modelled[rsc_type] == pytest.approx(resource_actual[rsc_type], abs=ABS_TOL, rel=REL_TOL), \
                f"Resource {rsc_type} does not match. Modelled: {resource_modelled[rsc_type]}, Actual: {resource_actual[rsc_type]}"


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
    def test_updating_properties(self, layer: LayerBase, config: dict):

        # updating coarse in
        coarse_in = max(layer.get_coarse_in_feasible())
        layer.coarse_in = coarse_in
        assert layer.coarse_in == coarse_in, f"Coarse in does not match. Expected: {coarse_in}, Actual: {layer.coarse_in}"

        # updating coarse out
        coarse_out = max(layer.get_coarse_out_feasible())
        layer.coarse_out = coarse_out
        assert layer.coarse_out == coarse_out, f"Coarse out does not match. Expected: {coarse_out}, Actual: {layer.coarse_out}"


