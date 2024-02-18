import glob
import random
import os
import unittest
import ddt
import json
import copy
import itertools
import numpy as np
import pytest

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fpgaconvnet.models.layers import LayerBase
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY

from pymongo import MongoClient
from pymongo.server_api import ServerApi

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
    def test_piecewise_rate_out(self, layer: LayerBase, config: dict): # TODO
        pass
        # # check piecewise rate out
        # for i in range(layer.ports_out):
        #     self.assertTrue(layer.piecewise_rate_out(i) >= 0.0)
        #     self.assertTrue(layer.piecewise_rate_out(i) <= 1.0)


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
    def test_updating_coarse_in(self, layer: LayerBase, config: dict):

        # copy the layer
        new_layer = copy.deepcopy(layer)

        # updating coarse in
        coarse_in = max(new_layer.get_coarse_in_feasible())
        new_layer.coarse_in = coarse_in
        assert new_layer.coarse_in == coarse_in, f"Coarse in does not match. Expected: {coarse_in}, Actual: {new_layer.coarse_in}"

        # check the latency either improves or stays the same
        assert new_layer.latency() <= layer.latency(), f"Latency did not improve. Expected: {new_layer.latency()}, Actual: {layer.latency()}"

        # check the resources either improve or stay the same
        for rsc_type in RESOURCES:
            assert new_layer.resource()[rsc_type] <= layer.resource()[rsc_type], \
                f"Resource {rsc_type} did not improve. Expected: {new_layer.resource()[rsc_type]}, Actual: {layer.resource()[rsc_type]}"

    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_updating_coarse_out(self, layer: LayerBase, config: dict):

        # copy the layer
        new_layer = copy.deepcopy(layer)

        # updating coarse out
        coarse_out = max(new_layer.get_coarse_out_feasible())
        new_layer.coarse_out = coarse_out
        assert new_layer.coarse_out == coarse_out, f"Coarse out does not match. Expected: {coarse_out}, Actual: {new_layer.coarse_out}"

        # check the latency either improves or stays the same
        assert new_layer.latency() <= layer.latency(), f"Latency did not improve. Expected: {new_layer.latency()}, Actual: {layer.latency()}"

        # check the resources either improve or stay the same
        for rsc_type in RESOURCES:
            assert new_layer.resource()[rsc_type] <= layer.resource()[rsc_type], \
                f"Resource {rsc_type} did not improve. Expected: {new_layer.resource()[rsc_type]}, Actual: {layer.resource()[rsc_type]}"


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_updating_coarse_out(self, layer: LayerBase, config: dict):

        if "convolution" in layer.name:

            # copy the layer
            new_layer = copy.deepcopy(layer)

            # updating coarse group
            coarse_group = max(new_layer.get_coarse_group_feasible())
            new_layer.coarse_group = coarse_group
            assert new_layer.coarse_group == coarse_group, f"Coarse group does not match. Expected: {coarse_group}, Actual: {new_layer.coarse_group}"

            # check the latency either improves or stays the same
            assert new_layer.latency() <= layer.latency(), f"Latency did not improve. Expected: {new_layer.latency()}, Actual: {layer.latency()}"

            # check the resources either improve or stay the same
            for rsc_type in RESOURCES:
                assert new_layer.resource()[rsc_type] <= layer.resource()[rsc_type], \
                    f"Resource {rsc_type} did not improve. Expected: {new_layer.resource()[rsc_type]}, Actual: {layer.resource()[rsc_type]}"


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_updating_fine(self, layer: LayerBase, config: dict):

        if "convolution" in layer.name:

            # copy the layer
            new_layer = copy.deepcopy(layer)

            # updating fine
            fine = max(new_layer.get_fine_feasible())
            new_layer.fine = fine
            assert new_layer.fine == fine, f"Fine does not match. Expected: {fine}, Actual: {new_layer.fine}"

            # check the latency either improves or stays the same
            assert new_layer.latency() <= layer.latency(), f"Latency did not improve. Expected: {new_layer.latency()}, Actual: {layer.latency()}"

            # check the resources either improve or stay the same
            for rsc_type in RESOURCES:
                assert new_layer.resource()[rsc_type] <= layer.resource()[rsc_type], \
                    f"Resource {rsc_type} did not improve. Expected: {new_layer.resource()[rsc_type]}, Actual: {layer.resource()[rsc_type]}"


    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_double_buffered_attribute_exists(self, layer: LayerBase, config: dict):

        # check the double buffered attribute
        assert hasattr(layer, "double_buffered")

    @ddt.unpack
    @ddt.named_data(*LAYERS)
    def test_change_stream_weights(self, layer: LayerBase, config: dict):

        if "convolution" in layer.name:

            # copy the layer
            new_layer = copy.deepcopy(layer)

            # randomly choose a ratio
            weight_step_size = random.choice([0.1, 0.2, 0.3, 0.4, 0.5,
                                            0.6, 0.7, 0.8, 0.9, 1.0])
            new_layer.stream_weights = new_layer.stream_unit() * \
                new_layer.stream_step(weight_step_size)

