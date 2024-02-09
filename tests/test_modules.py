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

from fpgaconvnet.models.modules import Fork, Accum, Glue, SlidingWindow, Conv, Pool, Squeeze, ReLU, Pad, Concat, GlobalPool

from pymongo import MongoClient
from pymongo.server_api import ServerApi

BACKEND = "chisel"
RESOURCES = ["LUT", "FF", "BRAM", "DSP"]

SERVER_DB="mongodb+srv://fpgaconvnet.hwnxpyo.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"

# absolute and relative tolerance
ABS_TOL = 200
REL_TOL = 0.05

def load_module_configs_db():
    """
    Load module configurations from the MongoDB database
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
    configs = collection.find({"hierarchy": "module"})

    # return the configurations
    return list(configs)


def initialise_modules(configs):
    """
    Initialise the modules from the configurations
    """

    # list of modules
    modules = []

    # initialise the modules
    for config in configs:

        # print(config)
        # add dimensions if not present
        config["rows"] = config.get("rows", 1)
        config["cols"] = config.get("cols", 1)
        config["channels"] = config.get("channels", 1)

        # other parameters
        config["groups"] = config.get("groups", 1)


        # build the module
        match config["name"]:
            case "fork":
                module = Fork(config["rows"], config["cols"], config["channels"],
                    config["fine"], config["coarse"], backend=BACKEND)
            case "accum":
                module = Accum(config["rows"], config["cols"], config["channels"],
                        config["filters"], config["groups"], backend=BACKEND)
            case "conv":
                module = Conv(config["rows"], config["cols"], config["channels"],
                        config["filters"], config["fine"], config["kernel_size"],
                        config["group"], backend=BACKEND)
            case "glue":
                module = Glue(config["rows"], config["cols"], config["channels"],
                        config["channels"], config["coarse"], 1, 1, backend=BACKEND)
            case "sliding_window":
                module = SlidingWindow(config["rows"], config["cols"], config["channels"],
                        config["kernel_size"], config["stride"], 0, 0, 0, 0, backend=BACKEND)
            case "pool":
                module = Pool(config["rows"], config["cols"], config["channels"],
                        config["kernel_size"], backend=BACKEND)
            case "pad":
                module = Pad(config["rows"], config["cols"], config["channels"],
                            config["pad_top"], config["pad_bottom"], config["pad_left"],
                            config["pad_right"], backend=BACKEND)
            case "squeeze":
                module = Squeeze(config["rows"], config["cols"], config["channels"],
                            config["coarse_in"], config["coarse_out"], backend=BACKEND)
            case "relu":
                module = ReLU(config["rows"], config["cols"], config["channels"],
                            backend=BACKEND)
            case "global_pool":
                module = GlobalPool(config["rows"], config["cols"], config["channels"],
                                    backend=BACKEND)

            case _:
                pass

        # create a name for the module
        name = f"module:{BACKEND}:2D:{config['name']} {config['_id']}"

        # append the module
        modules.append((name, module, config))

    # return the modules
    return modules

CONFIGS=load_module_configs_db()
MODULES=initialise_modules(CONFIGS)

@ddt.ddt
class TestModule(unittest.TestCase):

    @ddt.unpack
    @ddt.named_data(*MODULES)
    def test_rates(self, module, config: dict):

        # check rate in
        if isinstance(module, Concat):
            for i in range(module.ports_in):
                self.assertGreaterEqual(module.rate_in(i), 0.0)
                self.assertLessEqual(module.rate_in(i), 1.0)
        else:
            self.assertGreaterEqual(module.rate_in(), 0.0)
            self.assertLessEqual(module.rate_in(), 1.0)

        # check rate out
        self.assertGreaterEqual(module.rate_out(), 0.0)
        self.assertLessEqual(module.rate_out(), 1.0)


    # @ddt.unpack
    # @ddt.named_data(*MODULES)
    # def test_config_gen(self, module, config):

    #     # generate the config
    #     config_new = module.module_info()
    #     self.assertTrue(isinstance(config_new, dict))

    #     # get the module construction parameters
    #     name = config_new["name"]
    #     params = config_new
    #     backend = BACKEND[config_new["backend"]]
    #     dimensionality = DIMENSIONALITY(config_new["dimensionality"][0])

    #     # build from the config
    #     module = ModuleBase.build(name, params, backend, dimensionality)
    #     self.assertTrue(isinstance(module, type(module)))


    @ddt.unpack
    @ddt.named_data(*[  (f"{name} ({rsc_type})", rsc_type, module, config) \
            for rsc_type, (name, module, config) in itertools.product(RESOURCES, MODULES) ])
    def test_resources(self, rsc_type, module, config):

        # get the resource model
        resource = config["resource"]

        # check the resources
        modelled_rsc = module.rsc()[rsc_type]
        actual_rsc = resource.get(rsc_type, modelled_rsc)
        assert modelled_rsc >= 0
        assert modelled_rsc == pytest.approx(actual_rsc, abs=ABS_TOL, rel=REL_TOL), \
            f"Resource {rsc_type} does not match. Modelled: {modelled_rsc}, Actual: {actual_rsc}"


    @ddt.unpack
    @ddt.named_data(*MODULES)
    def test_cycles(self, module, config):

        # get the cycles
        cycles = config["cycles"]

        # get the modelled cycles
        model_cycles = module.latency()

        # check the cycles
        if cycles > 0:
            assert model_cycles == pytest.approx(cycles, abs=ABS_TOL, rel=REL_TOL), \
                f"Modelled cycles do not match. Expected: {cycles}, Actual: {model_cycles}"



