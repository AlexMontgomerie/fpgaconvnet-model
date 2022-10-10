import unittest
import ddt

from fpgaconvnet.optimiser.simulated_annealing import SimulatedAnnealing
from fpgaconvnet.optimiser.improve import Improve

class TestOptimiserLeNetSimulatedAnnealing(unittest.TestCase):

    def setUp(self):

        # setup optimiser configuration
        T=10.0
        T_min=0.01
        k=0.01
        cool=0.9
        iterations=10

        # setup optimiser
        self.net = SimulatedAnnealing("lenet", "examples/models/lenet.onnx",
            T=T, T_min=T_min, k=k, cool=cool, iterations=iterations)

        # turn on debugging
        self.net.DEBUG = True

        # update platform information
        self.net.update_platform("examples/platforms/zc706.json")

        # specify optimiser objective
        self.net.objective  = 1

        # completely partition graph
        self.net.split_complete()

        # apply complete max weights reloading
        for partition_index in range(len(self.net.partitions)):
            self.net.partitions[partition_index].apply_max_weights_reloading()

    def test_running_optimiser(self):

        # get throughput before
        throughput_prev = self.net.get_throughput()

        # run optimiser
        self.net.run_optimiser()

        # get throughput after
        throughput = self.net.get_throughput()

        # check optimiser gives greater throughput
        self.assertGreaterEqual(throughput, throughput_prev)
