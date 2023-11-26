"""
A command line interface for running the optimiser for given networks
"""

import os
import toml
import argparse
import random

from tabulate import tabulate
from tqdm import tqdm
from dacite import from_dict

from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model
from fpgaconvnet.models.modules.resources import SVRResourceModel, NNLSHeuristicResourceModel, NNLSResourceModel
from fpgaconvnet.models.modules.database import get_database, Record
from fpgaconvnet.platform import build_platform_from_toml

def main():

    # get all the module names
    module_names = set([ m.name for m in ModuleBase.MODULE_REGISTRY.values() ])

    parser = argparse.ArgumentParser(description="Command line interface for generating hardware resource models")
    parser.add_argument("-n", "--name", type=str, required=True, help=f"name of the resource modelling run")
    parser.add_argument('-m','--module', type=str, required=True, choices=list(module_names), help=f"target module")
    parser.add_argument('-b','--backend', type=str, required=True, choices=["chisel", "hls"], help="target backend for resource modelling")
    parser.add_argument('-d','--dimensionality', metavar="INT", type=int, required=True, choices=[2,3], help="dimensionality of the target module")
    parser.add_argument('-c','--config-path', metavar='PATH', required=True, help="path for the resource modelling configuration file (.toml)")
    parser.add_argument('-p','--platform-path', metavar='PATH', required=True, help="path to the platform description file (.toml)")
    parser.add_argument('-l','--db-limit', metavar='INT', type=int, default=10000, help="limit the number of records to use from the database")
    parser.add_argument("-s", "--test-split", type=float, default=0.2, help="fraction of the dataset to use for testing")

    # parse the arguments
    args = parser.parse_args()

    # load the configuration file
    config = toml.load(args.config_path)

    # load the platform
    platform = build_platform_from_toml(args.platform_path)

    # get the module class
    backend = BACKEND[args.backend.upper()]
    dimensionality = DIMENSIONALITY(args.dimensionality)
    module_class = ModuleBase.get_all_modules(args.module, backend, dimensionality)[0]

    # get the database information
    db = get_database()
    collection = db[config["database"]["collection"]]

    # filter for the relevant records
    filters = {
        "config.name": args.module,
        "config.backend": args.backend.upper(),
        **config["database"]["filters"],
        **platform.get_resource_database_filters(),
    }
    collection = collection.find(filters).limit(args.db_limit)

    # get the training and testing data
    dataset = list(tqdm(collection, desc="loading points from database"))

    # split the dataset
    random.shuffle(dataset)
    split_idx = int(len(dataset) * args.test_split)
    train_data = Record(module=module_class, docs=dataset[split_idx:])
    test_data = Record(module=module_class, docs=dataset[:split_idx])

    for rsc_type in platform.resource_types:

        # get the model
        config["model"]["module_type"] = module_class
        config["model"]["rsc_type"] = rsc_type
        match config["model"]["type"]:
            case "SVR":
                model = from_dict(data_class=SVRResourceModel, data=config["model"])
            case "NNLS":
                model = from_dict(data_class=NNLSResourceModel, data=config["model"])
            case "NNLSHeuristic":
                model = from_dict(data_class=NNLSHeuristicResourceModel, data=config["model"])
            case _:
                raise NotImplementedError(f"Unsupported model type: {config['model']['type']}")

        # fit the model
        model.fit(train_data)

        # get the accuracy
        accuracy = model.get_accuracy(test_data)
        print(tabulate([accuracy], headers="keys"))

        # cache the model locally
        filename = f"{args.name}.{rsc_type}.{args.module}.{args.backend}.{args.dimensionality}.model"
        cache_path = os.path.join(os.path.dirname(__file__), "..", "cache", "modules", filename)
        model.save_model(cache_path)
        print(f"Saved model to {cache_path}\n")

if __name__ == "__main__":
    main()
