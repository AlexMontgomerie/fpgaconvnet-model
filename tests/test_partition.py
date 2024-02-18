import unittest
import ddt
import pickle
import itertools
import pytest

import onnx

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from fpgaconvnet.tools.layer_enum import LAYER_TYPE
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.models.partition.partition import Partition
import fpgaconvnet.tools.graphs as graphs
import networkx as nx

from fpgaconvnet.parser.parser import Parser

from pymongo import MongoClient
from pymongo.server_api import ServerApi

RESOURCES = ["LUT", "FF", "BRAM", "DSP"]

SERVER_DB="mongodb+srv://fpgaconvnet.hwnxpyo.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"

# absolute and relative tolerance
ABS_TOL = 200
REL_TOL = 0.05

def load_partition_configs_db():
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
    configs = collection.find({"hierarchy": "partition"})

    # return the configurations
    return list(configs)

def initialise_partitions(configs):
    """
    Initialise the modules from the configurations
    """

    # list of modules
    partitions = []

    # initialise the modules
    for config in configs:

        # deserialise the onnx model
        onnx_model = pickle.loads(config["model"])

        # check the onnx model is valid
        onnx.checker.check_model(onnx_model)

        # save the onnx model temporarily
        network_path = os.path.join("/tmp", "temp.onnx")
        onnx.save_model(onnx_model, network_path)

        # convert the onnx model to fpgaconvnet network
        net = Parser(backend=BACKEND.CHISEL).onnx_to_fpgaconvnet(network_path, save_opt_model=False)

        # create an instance of the partition hardware
        partition = Partition(net.partitions[0].graph,
                        Architecture(backend=BACKEND.CHISEL,
                                     dimensionality=DIMENSIONALITY.TWO))
        # create a name for the module
        name = f"partition:{config['name']} {config['_id']}"

        # append the module
        partitions.append((name, partition, config))

    # return the modules
    return partitions

CONFIGS=load_partition_configs_db()
PARTITIONS=initialise_partitions(CONFIGS)

@ddt.ddt
class TestPartition(unittest.TestCase):

    @ddt.unpack
    @ddt.named_data(*PARTITIONS)
    def test_validation(self, parition: Partition, config: dict): pass
        # run all validation checks
        # network.check_ports()
        # network.check_workload()
        # network.check_streams()
        # network.check_partitions()
        # network.check_memory_bandwidth()


    @ddt.unpack
    @ddt.named_data(*PARTITIONS)
    def test_save_all_partitions(self, parition: Partition, config: dict):
        # network.save_all_partitions("/tmp/fpgaconvnet-test-network-config.json")
        assert isinstance(parition.partition_info_dict(), dict)


    @ddt.unpack
    @ddt.named_data(*[  (f"{name} ({rsc_type})", rsc_type, layer, config) \
            for rsc_type, (name, layer, config) in itertools.product(RESOURCES, PARTITIONS) ])
    def test_resources(self, rsc_type, parition, config):

        # check the resources
        actual_rsc= config["resource"][rsc_type]
        modelled_rsc= parition.get_resource_usage()[rsc_type]

        assert modelled_rsc >= 0
        assert modelled_rsc == pytest.approx(actual_rsc, abs=ABS_TOL, rel=REL_TOL), \
            f"Resource {rsc_type} does not match. Modelled: {modelled_rsc}, Actual: {actual_rsc}"


    @ddt.unpack
    @ddt.named_data(*PARTITIONS)
    def test_cycles(self, parition, config):

        # get the cycles
        cycles = config["cycles"]

        # get the modelled cycles
        model_cycles = parition.get_cycle()

        # check the cycles
        if cycles > 0:
            assert model_cycles == pytest.approx(cycles, abs=ABS_TOL, rel=REL_TOL), \
                f"Modelled cycles do not match. Expected: {cycles}, Actual: {model_cycles}"


    @ddt.unpack
    @ddt.named_data(*PARTITIONS)
    def test_remove_squeeze(self, parition: Partition, config: dict):

        # perform the remove squeeze operation
        parition.remove_squeeze()

        # check the graph
        for node in parition.graph.nodes:
            assert parition.graph.nodes[node]["type"] != LAYER_TYPE.Squeeze, \
                "Squeeze node not removed"



