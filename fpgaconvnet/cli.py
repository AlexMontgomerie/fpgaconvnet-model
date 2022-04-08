"""
A command line interface for running the optimiser for given networks
"""

import logging
import os
import yaml
import json
import argparse
import shutil
import random
import numpy as np

from fpgaconvnet_optimiser.optimiser.simulated_annealing import SimulatedAnnealing
from fpgaconvnet_optimiser.optimiser.improve import Improve
from fpgaconvnet_optimiser.optimiser.greedy_partition import GreedyPartition

import fpgaconvnet_optimiser.tools.graphs as graphs
from fpgaconvnet_optimiser.tools.layer_enum import from_onnx_op_type

def main():
    parser = argparse.ArgumentParser(description="fpgaConvNet Optimiser Command Line Interface")
    parser.add_argument('-n','--name', metavar='PATH', required=True,
        help='network name')
    parser.add_argument('-m','--model_path', metavar='PATH', required=True,
        help='Path to ONNX model')
    parser.add_argument('-p','--platform_path', metavar='PATH', required=True,
        help='Path to platform information')
    parser.add_argument('-o','--output_path', metavar='PATH', required=True,
        help='Path to output directory')
    parser.add_argument('-b','--batch_size', metavar='N',type=int, default=1, required=False,
        help='Batch size')
    parser.add_argument('--objective', choices=['throughput','latency'], required=True,
        help='Optimiser objective')
    parser.add_argument('--optimiser', choices=['simulated_annealing', 'improve', 'greedy_partition'],
        default='improve', help='Optimiser strategy')
    parser.add_argument('--optimiser_config_path', metavar='PATH', required=True,
        help='Configuration file (.yml) for optimiser')
    parser.add_argument('--teacher_partition_path', metavar='PATH', required=False,
        help='Previously optimised partitions saved in JSON')
    parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
        help='Seed for the optimiser run')

    # parse the arguments
    args = parser.parse_args()

    # setup seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # make the output directory if it does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # copy input files to the output path
    shutil.copy(args.model_path, os.path.join(args.output_path,os.path.basename(args.model_path)) )
    shutil.copy(args.platform_path, os.path.join(args.output_path,os.path.basename(args.platform_path)) )

    # load optimiser config
    with open(args.optimiser_config_path,"r") as f:
        optimiser_config = yaml.load(f, Loader=yaml.Loader)

    # Initialise logger
    if bool(optimiser_config["general"]["logging"]):
        FORMAT="%(asctime)s.%(msecs)03d %(levelname)s = (%(module)s) %(message)s"
        logging.basicConfig(level=logging.INFO, filename=os.path.join(args.output_path,"optimiser.log"), format=FORMAT, filemode="w", datefmt='%H:%M:%S')
    else:
        logging.getLogger().disabled = True

    # create the checkpoint directory
    if not os.path.exists(os.path.join(args.output_path,"checkpoint")):
        os.makedirs(os.path.join(args.output_path,"checkpoint"))

    # format the partition transform allowed partitions
    allowed_partitions = []
    for allowed_partition in optimiser_config["transforms"]["partition"]["allowed_partitions"]:
        allowed_partitions.append((from_onnx_op_type(allowed_partition[0]), from_onnx_op_type(allowed_partition[1])))
    optimiser_config["transforms"]["partition"]["allowed_partitions"] = allowed_partitions

    # load network based on the given optimiser strategy
    if args.optimiser == "improve":
        net = Improve(args.name,args.model_path,
                T=float(optimiser_config["annealing"]["T"]),
                T_min=float(optimiser_config["annealing"]["T_min"]),
                k=float(optimiser_config["annealing"]["k"]),
                cool=float(optimiser_config["annealing"]["cool"]),
                iterations=int(optimiser_config["annealing"]["iterations"]),
                transforms_config=optimiser_config["transforms"])
    elif args.optimiser == "simulated_annealing":
        net = SimulatedAnnealing(args.name,args.model_path,
                T=float(optimiser_config["annealing"]["T"]),
                T_min=float(optimiser_config["annealing"]["T_min"]),
                k=float(optimiser_config["annealing"]["k"]),
                cool=float(optimiser_config["annealing"]["cool"]),
                iterations=int(optimiser_config["annealing"]["iterations"]),
                transforms_config=optimiser_config["transforms"],
                checkpoint=bool(optimiser_config["general"]["checkpoints"]),
                checkpoint_path=os.path.join(args.output_path,"checkpoint"))
    elif optimiser == "greedy_partition":
        net = GreedyPartition(name, model_path,
                T=float(optimiser_config["annealing"]["T"]),
                T_min=float(optimiser_config["annealing"]["T_min"]),
                k=float(optimiser_config["annealing"]["k"]),
                cool=float(optimiser_config["annealing"]["cool"]),
                iterations=int(optimiser_config["annealing"]["iterations"]),
                transforms_config=optimiser_config["transforms"])

    # update the resouce allocation
    net.rsc_allocation = float(optimiser_config["general"]["resource_allocation"])

    # turn on debugging
    net.DEBUG = True

    # get platform
    with open(args.platform_path,'r') as f:
        platform = json.load(f)

    # update platform information
    net.update_platform(args.platform_path)

    # specify optimiser objective
    if args.objective == "throughput":
        net.objective  = 1
    if args.objective == "latency":
        net.objective  = 0

    # specify batch size
    net.batch_size = args.batch_size

    # specify available transforms
    net.get_transforms()

    # initialize graph
    ## completely partition graph
    if bool(optimiser_config["transforms"]["partition"]["start_complete"]):
        net.split_complete()

    ## apply max fine factor to the graph
    if bool(optimiser_config["transforms"]["fine"]["start_complete"]):
        for partition in net.partitions:
            partition.apply_complete_fine()

    ## apply complete max weights reloading
    if bool(optimiser_config["transforms"]["weights_reloading"]["start_max"]):
        for partition_index in range(len(net.partitions)):
            net.partitions[partition_index].apply_max_weights_reloading()

    if bool(optimiser_config["general"]["starting_point_distillation"]):
        net.update_partitions()
        net.starting_point_distillation(args.teacher_partition_path)
        net.update_partitions()
        net.merge_memory_bound_partitions()
        net.update_partitions()

    # run optimiser
    net.run_optimiser()

    # update all partitions
    net.update_partitions()

    # find the best batch_size
    #if args.objective == "throughput":
    #    net.get_optimal_batch_size()

    # visualise network
    net.visualise(os.path.join(args.output_path,"topology.png"))

    # create report
    net.create_report(os.path.join(args.output_path,"report.json"))

    # save all partitions
    net.save_all_partitions(args.output_path)

    # create scheduler
    net.get_schedule_csv(os.path.join(args.output_path,"scheduler.csv"))
