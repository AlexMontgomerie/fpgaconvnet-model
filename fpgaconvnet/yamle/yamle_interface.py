import onnx
import torch
import numpy as np
import logging

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.optimiser.solvers import Improve
from fpgaconvnet.optimiser.solvers import SimulatedAnnealing
from fpgaconvnet.optimiser.solvers import GreedyPartition

import fpgaconvnet.optimiser.transforms.partition
import fpgaconvnet.optimiser.transforms.coarse
import fpgaconvnet.optimiser.transforms.fine

from typing import Dict

log = logging.getLogger("pytorch_lightning")

def _onnx_is_empty(path: str) -> bool:
    graph = onnx.load(path).graph
    return len(graph.node) == 0

def get_yamle_model_cost(model_path: str, platform_path: str) -> Dict[str, float]:
    # check model is not empty
    # assert(not _onnx_is_empty(model_path), "Error: Got an empty onnx graph.")
    if _onnx_is_empty(model_path):
        log.warning("Warning: Got an empty graph \"{}\".".format(model_path))
        return {
            "latency": float("nan"),
            "throughput": float("nan")
        }

    # create a parser
    parser = Parser(backend="chisel", quant_mode="auto", convert_gemm_to_conv=False, custom_onnx=True)

    # get the network
    net = parser.onnx_to_fpgaconvnet(model_path, platform_path, save_opt_model=False)

    # greedy optimiser
    opt = GreedyPartition(net)

    # set latency objective
    opt.objective  = 0

    # set only fine and coarse transforms
    opt.transforms = ["fine", "coarse"]

    # disable weights reloading
    for partition in opt.net.partitions:
        partition.enable_wr = False

    # apply max fine factor
    for partition in net.partitions:
        fpgaconvnet.optimiser.transforms.fine.apply_complete_fine(partition)

    # update network
    opt.net.update_partitions()

    # run optimiser
    opt.run_solver()

    # update all partitions
    opt.net.update_partitions()
    opt.merge_memory_bound_partitions()
    opt.net.update_partitions()

    ## update buffer depths
    for node in opt.net.partitions[0].graph.nodes:
        if opt.net.partitions[0].graph.nodes[node]["type"] \
                in [ LAYER_TYPE.EltWise, LAYER_TYPE.Concat ]:
            opt.net.partitions[0].update_multiport_buffer_depth(node)

    report = {
        "latency" : opt.net.get_latency(fast=False),
        "throughput" : opt.net.get_throughput(),
        # "performance" : total_operations/self.get_latency(),
        # "cycles" : self.get_cycle(),
        # "partition_delay" : self.get_partition_delay()
        # "LUT" : torch.tensor(int(np.sum([ partition.get_resource_usage()["LUT"] for partition in opt.net.partitions ]))),
        # "FF" : torch.tensor(int(np.sum([ partition.get_resource_usage()["FF"] for partition in opt.net.partitions ]))),
        # "BRAM" : torch.tensor(int(np.sum([ partition.get_resource_usage()["BRAM"] for partition in opt.net.partitions ]))),
        # "DSP" : torch.tensor(int(np.sum([ partition.get_resource_usage()["DSP"] for partition in opt.net.partitions ])))
    }

    return report

    # # create report
    # opt.net.create_report(os.path.join(output_path,"report.json"))

    # # save all partitions
    # opt.net.save_all_partitions(os.path.join(output_path, "config.json"))

    # # create scheduler
    # opt.net.get_schedule_csv(os.path.join(output_path,"scheduler.csv"))