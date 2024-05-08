import os
import random
import toml
from tqdm import tqdm
from tabulate import tabulate
from dacite import from_dict
from typing import ClassVar, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from platformdirs import user_cache_dir
from pathvalidate import sanitize_filename

from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model
from fpgaconvnet.models.modules.resources import SVRResourceModel, NNLSHeuristicResourceModel, NNLSResourceModel
from fpgaconvnet.models.modules.database import get_database, Record

from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

@dataclass
class PlatformBase:

    # platform specific fields
    part: str
    """part identifier for the synthesis tool"""
    resources: dict[str,int]
    reconf_time: float
    """reconfiguration time in seconds"""
    freq: float = 100.0
    """board frequency in MHz"""
    mem_bw: float = 10.0
    port_width: int = 512

    # FPGA family specific fields
    family: ClassVar[str]
    resource_types: ClassVar[list[str]]

    # database specific fields
    db_collection: str = "latest"
    db_limit: int = 10000

    # resource model specific fields
    model_type: str = "NNLSHeuristic"
    model_test_split: float = 0.2

    # flag to build resource models if not already built
    build_models_if_doesnt_exist: bool = False


    def __post_init__(self):
        """
        Perform post initialisation checks.
        """
        # check resource types
        for rsc in self.resources.keys():
            assert rsc in self.resource_types, \
                    f"resource type {rsc} must be one of {self.resource_types}"

        # check resource values
        assert all([rsc_val >= 0 for rsc_val in self.resources.values()]), \
                "resource values must be positive"

        # check reconfiguration time
        assert self.reconf_time >= 0, "reconfiguration time must be positive"

        # check board frequency
        assert self.freq > 0, "board frequency must be positive"

        # check memory bandwidth
        assert self.mem_bw > 0, "memory bandwidth must be positive"

        # check port width
        assert self.port_width > 0, "port width must be positive"

        # create an empty dictionary to cache loaded resource models
        self.resource_models: dict[str,dict[str,ResourceModel]] = \
                { m: {rsc_type: None for rsc_type in self.resource_types} \
                for m in ModuleBase.MODULE_REGISTRY.keys() }

        # create a cache directory for the models
        self.model_cache_dir = os.path.join(user_cache_dir("fpgaconvnet", "models"),
                                            sanitize_filename(self.part))
        os.makedirs(self.model_cache_dir, exist_ok=True)

    @classmethod
    def from_toml(cls, platform_path: str):
        """
        Initialise platform from a TOML configuration file.
        An example of the format for this TOML file is shown below:

        ```toml
        [device]
        name = "zedboard"
        part = "xc7z020clg484-1"
        board = "xilinx.com:zc702:part0:1.4"

        [resources]
        DSP   = 220
        BRAM  = 280
        FF    = 106400
        LUT   = 53200

        [system]
        board_frequency = 200 # in MHz
        memory_bandwidth = 100.0 # in Gbps
        reconfiguration_time = 0.08255 # in seconds
        port_width = 128
        ```

        Args:
            platform_path: path to the platform configuration file

        Returns:
            platform object initialised from the configuration file
        """
        # make sure toml configuration
        assert os.path.splitext(platform_path)[1] == ".toml", "must be a TOML configuration file"

        # parse platform configuration toml file
        with open(platform_path, "r") as f:
            conf = toml.load(f)

        # initialise platform
        return cls(
            part=conf["device"]["part"],
            resources=conf["resources"],
            reconf_time=conf["system"]["reconfiguration_time"],
            freq=conf["system"]["board_frequency"],
            mem_bw=conf["system"]["memory_bandwidth"],
            port_width=conf["system"]["port_width"]
        )


    def get_resource(self, rsc_type: str) -> int:
        """
        Gets the number of resources for the specified resource type.

        Args:
            rsc_type: resource type

        Returns:
            number of resources, as an integer
        """
        if rsc_type not in self.resource_types:
            raise ValueError(f"resource type {rsc_type} not supported by family {self.family} (should be one of {self.resource_types})")
        return self.resources[rsc_type]


    def update_from_toml(self, platform_path: str):
        """
        Update platform configuration from a TOML configuration file.

        Args:
            platform_path: path to the platform configuration file
        """
        # make sure toml configuration
        assert os.path.splitext(platform_path)[1] == ".toml", "must be a TOML configuration file"

        # parse platform configuration toml file
        with open(platform_path, "r") as f:
            conf = toml.load(f)

        # update fields
        self.part = conf["device"]["part"]

        ## resources
        for resource, val in conf["resources"].items():
            self.resources[resource] = val

        ## system
        self.freq  = conf["system"].get("board_frequency", 100.0) # in MHz
        self.mem_bw      = conf["system"].get("memory_bandwidth", 5.0) # in Gbps
        self.reconf_time = conf["system"].get("reconfiguration_time", 0.0) # in seconds
        self.port_width  = conf["system"].get("port_width", 512) # in bits

        ## ethernet
        if "ethernet" in conf.keys():
            self.eth_bw = conf["ethernet"]["bandwidth"] # in Gbps
            self.eth_port_width = self.calculate_eth_port_width() # in bits
            self.eth_delay = conf["ethernet"]["latency"] # in seconds

        # perform post initialisation again
        self.__post_init__()


    def calculate_eth_port_width(self):
        """
        Calculate the equivalent Ethernet port width from the asynchronous FIFO.
        """
        # equivalent ethernet port width from async fifo
        mac_head = 10  # Bytes
        ip_head = 20  # Bytes
        udp_head = 8  # Bytes
        max_packet_size = 576  # Bytes, reassembly buffer size

        # todo: apply the constraint as bandwidth instead of port width
        eff_bw = self.eth_bw * (max_packet_size - mac_head - ip_head - udp_head) / max_packet_size
        eth_port_width = eff_bw / (self.freq / 1000)  # in bits
        return int(eth_port_width)


    def get_model_cache_string(self, module: ModuleBase, rsc_type: str) -> str:
        """
        Get the cache string for the resource model.

        Args:
            module: module object
            rsc_type: resource type
        """
        backend = module.backend.name.lower()
        dimensionality = min(module.dimensionality).value
        return f"{self.family}.{module.name}.{backend}.{dimensionality}.{rsc_type}"


    def get_resource_database_filters(self, scale: float = 0.8) -> dict[str,Union[str,dict[str,int]]]:
        """
        Gets the filters needed to query the database for resource runs
        relevant to the platform.

        Args:
            scale: scale factor to apply to the resources

        Returns:
            dictionary of filters
        """
        filters = { "fpga": { "$regex": f"{self.family}.*" } }
        for rsc_type, rsc_max in self.resources.items():
            filters[f"resource.{rsc_type}"] = { "$exists": True, "$lt": int(scale*self.get_resource(rsc_type)) }
        return filters


    def build_resource_models(self, module: ModuleBase):
        """
        Build resource models for the platform.
        This method will build resource models for the specified module, backend and dimensionality.
        The model is saved to the cache directory, which is specified by the `model_cache_dir` field.
        The file name is constructed as follows:
            `{family}.{module}.{backend}.{dimensionality}.{rsc_type}.model`

        see the `get_model_cache_string` method for more details on the file name.

        Args:
            module: module object
        """
        # get the backend and dimensionality
        backend = module.backend
        dimensionality = min(module.dimensionality)

        # get the module class
        module_class = ModuleBase.get_all_modules(module.name, backend, dimensionality)[0]

        # get the database information
        db = get_database()
        collection = db[self.db_collection]

        # filter for the relevant records
        filters = {
            "config.name": module_class.name,
            "config.backend": module.backend.name.upper(),
            **self.get_resource_database_filters(),
        }
        collection = collection.find(filters).limit(self.db_limit)

        # get the training and testing data
        dataset = list(tqdm(collection, desc=f"loading points from database for {module_class.__name__}"))

        # check if the collection is empty
        if len(dataset) == 0:
            raise ValueError(f"no records found in the database for {module_class.__name__}")

        # split the dataset
        random.shuffle(dataset)
        split_idx = int(len(dataset) * self.model_test_split)
        train_data = Record(module=module_class, docs=dataset[split_idx:])
        test_data = Record(module=module_class, docs=dataset[:split_idx])

        # accuracy dictionary for logging
        accuracy = {}

        # iterate over the resource types
        for rsc_type in self.resource_types:

            # get the model configuration
            model_config = {
                "module_type": module_class,
                "rsc_type": rsc_type,
            }

            # get the model
            match self.model_type:
                case "SVR":
                    model = from_dict(data_class=SVRResourceModel, data=model_config)
                case "NNLS":
                    model = from_dict(data_class=NNLSResourceModel, data=model_config)
                case "NNLSHeuristic":
                    model = from_dict(data_class=NNLSHeuristicResourceModel, data=model_config)
                case _:
                    raise NotImplementedError(f"Unsupported model type: {self.model_type}")

            # fit the model
            model.fit(train_data)

            # get the accuracy
            accuracy[rsc_type] = model.get_accuracy(test_data)

            # save the model to the cache directory
            cache_path = os.path.join(self.model_cache_dir,
                    f"{self.get_model_cache_string(module, rsc_type)}.model")
            model.save_model(cache_path)
            print(f"Saved model for {module_class.__name__}:{rsc_type} to {cache_path}")

            # add the model to the resource_models dictionary
            self.resource_models[module_class.__name__][rsc_type] = model

        # return the accuracy
        return accuracy

        # # print a table with summary of the accuracy
        # table = [ [rsc_type, *list(acc.values())] for rsc_type, acc in accuracy.items() ]
        # header = [ "Resource", *list(list(accuracy.values())[0].keys()) ]
        # print(tabulate(table, headers=header) + "\n")


    def build_all_resource_models(self):
        """
        Build all resource models for the platform.
        """
        for module in ModuleBase.MODULE_REGISTRY.values():
            try: self.build_resource_models(module)
            # except ValueError as e: pass
            except Exception as e: print(f"Error building resource models for {module.__name__}: {e}")


    def load_resource_models(self, module: ModuleBase, rsc_type: str):
        """
        Load resource models for the platform.

        Args:
            module: module object
            rsc_type: resource type
        """

        # get the cache path
        cache_path = os.path.join(self.model_cache_dir,
                f"{self.get_model_cache_string(module, rsc_type)}.model")

        # check if the model exists
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"resource model for {module.class_name}:{rsc_type} not found at {cache_path}")

        # update the resource models
        self.resource_models[module.class_name][rsc_type] = ResourceModel.load_model(cache_path)


    def load_all_resource_models(self):
        """
        Load all resource models for the platform.
        """
        for module in ModuleBase.MODULE_REGISTRY.values():
            for rsc_type in self.resource_types:
                self.load_resource_models(module, rsc_type)


    def get_resource_model(self, module: ModuleBase, rsc_type: str):
        """
        Method to get the resource model for the specified module and resource type.
        If the resource model is not already loaded, it will be loaded from the cache directory.
        This expects that the resource model has been built using the `build_resource_models` method,
        using the configuration for this particular instance of the platform.

        Args:
            module: module object
            rsc_type: resource type

        Returns:
            resource model object
        """
        # load the resource model if it is not already loaded
        if self.resource_models[module.class_name][rsc_type] is None:
            self.load_resource_models(module, rsc_type)

        # return the resource model
        return self.resource_models[module.class_name][rsc_type]


