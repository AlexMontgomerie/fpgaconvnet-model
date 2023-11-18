import os
from dataclasses import dataclass
from tqdm import tqdm
from dacite import from_dict
from typing import Tuple, Dict
import random

from pymongo import MongoClient
from pymongo.server_api import ServerApi

from fpgaconvnet.models.modules import ModuleBase

SERVER_DB="mongodb+srv://fpgaconvnet.hwnxpyo.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"
LIMIT=20000

@dataclass(kw_only=True)
class Record:
    module: ModuleBase
    docs: list[dict]

    def build_module(self, config) -> ModuleBase:
        return from_dict(data_class=self.module, data=config)

    def modules(self) -> list[ModuleBase]:
        return [ self.build_module(d["config"]) for d in self.docs ]

    def parameters(self) -> list[list[int]]:
        modules = self.modules()
        return [ m.resource_parameters() for m in modules ]

    def heuristic_parameters(self, rsc_type: str) -> list[list[int]]:
        modules = self.modules()
        return [ m.resource_parameters_heuristics()[rsc_type] for m in modules ]

    def resources(self, rsc_type: str) -> list[int]:
        return [ d["resource"][rsc_type] for d in self.docs ]

    def power(self, pwr_type: str) -> list[float]:
        return [ d["power"][pwr_type] for d in self.docs ]

    def timing(self) -> list[float]:
        return [ d["timing"]["wns"] for d in self.docs ]

def get_database():

    # database .pem path
    db_pem = os.path.join(os.path.dirname(__file__),
            "fpgaconvnet-mongodb.pem")

    # create MongoDB client
    client = MongoClient(SERVER_DB, tls=True,
        tlsCertificateKeyFile=db_pem,
        server_api=ServerApi('1'))

    # return database
    return client["fpgaconvnet"]

def get_collection(collection_name: str):

    # get database
    db = get_database()

    # get collection
    return db[collection_name]

def get_modelling_collection(collection_name: str, module: ModuleBase):

    # TODO: add extra filters

    # get the collection
    collection = get_collection(collection_name)

    # filter by module name
    return collection.find({
        "config.type": "ACCUMHARDWARE", # TODO: remove this
        # "config.name": module.name,
        # "config.backend": module.backend,
    }).limit(LIMIT)


def load_resource_dataset(collection_name: str, module: ModuleBase,
        use_cache: bool = False, test_split: float = 0.2) -> Tuple[Record, Record]:

    # TODO: implement caching

    # get the collection
    collection = get_modelling_collection(collection_name, module)

    # convert collection to resource records

    # load the dataset
    dataset = list(tqdm(collection, desc="loading points from database"))

    # split the dataset
    random.shuffle(dataset)
    split_idx = int(len(dataset) * test_split)
    train_data = Record(module=module, docs=dataset[split_idx:])
    test_data = Record(module=module, docs=dataset[:split_idx])

    # return the train and test data
    return train_data, test_data

