"""

"""
import collections
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
import math

import numpy as np
import pydot
from dacite import from_dict
from google.protobuf.json_format import MessageToDict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.layers.utils import balance_module_rates, get_factors
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules import ModuleBase

class LayerBaseMeta(type, metaclass=ABCMeta):

    LAYER_REGISTRY = {}

    def __new__(cls, *args, **kwargs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = super().__new__(cls, *args, **kwargs)
        if new_cls.register:
            cls.LAYER_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.LAYER_REGISTRY)

    @classmethod
    def get_all_layers(cls, name: str, backend: str, dimensionality: int):

        # get all the modules in the registry
        modules = list(cls.LAYER_REGISTRY.values())

        # filter all the modules with the given name
        modules = list(filter(lambda m: m.name == name, modules))

        # filter all the modules with the given backend
        modules = list(filter(lambda m: m.backend == backend, modules))

        # filter all the modules with the given dimensionality
        modules = list(filter(lambda m: dimensionality == m.dimensionality, modules))

        return modules

    @classmethod
    def build(cls, name: str, config: dict, backend: str, dimensionality: int):

        # get all the relevant layers
        layers = cls.get_all_layers(name, backend, dimensionality)

        # check there is at least 1 module
        assert len(layers) > 0, f"No layers found for name={name}, \
                backend={backend.name}, dimensionality={dimensionality.value}"

        # check there is only a single module left
        assert len(layers) == 1, f"Too many layers found for name={name}, \
                backend={backend.name}, dimensionality={dimensionality.value}"

        # get the module class
        layer= layers[0]

        # create a new instance of the module
        return from_dict(data_class=layer, data=config)


    @classmethod
    def build_from_dict(cls, name: str, conf: dict):
        inst = from_dict(data_class=cls.LAYER_REGISTRY[name], data=conf)
        inst.__post_init__()
        return inst

@dataclass(kw_only=True)
class LayerBase(metaclass=LayerBaseMeta):

    coarse_in: int = 1
    coarse_out: int = 1
    mem_bw_in: float = 100.0
    mem_bw_out: float = 100.0
    data_t: FixedPoint = FixedPoint(16,8)
    modules: dict = field(default_factory=collections.OrderedDict, init=True)

    name: ClassVar[str]
    backend: ClassVar[BACKEND]
    dimensionality: ClassVar[set[DIMENSIONALITY]]
    register: ClassVar[bool] = False

    def __post_init__(self):

        self.input_t = self.data_t
        self.output_t = self.data_t
        self.stream_inputs = [False]
        self.stream_outputs = [False]
        self.is_init = True

        self.create_modules()

    def __setattr__(self, name, value):
        """
        Set the value of an attribute and update the layer.

        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to set the attribute to.

        Raises:
            AssertionError: If the value is not feasible for the attribute.

        Returns:
            None
        """

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        match name:
            case "coarse_in":
                assert(value in self.get_coarse_in_feasible())
                super().__setattr__(name, value)
                self.update()

            case "coarse_out":
                assert(value in self.get_coarse_out_feasible())
                super().__setattr__(name, value)
                self.update()

            case _:
                super().__setattr__(name, value)

    def create_modules(self):

        # create the module graph
        for name, config_fn in self.module_lookup.items():
            self.modules[name] = ModuleBase.build(
                    name, config_fn(), self.backend, self.dimensionality)

    # def build_module_graph(self):

    #     # initialise the digraph
    #     self.module_graph = nx.DiGraph()

    #     # create all the nodes
    #     for name, module in self.modules.items():
    #         self.module_graph.add_node(name, module=module)


    def update_modules(self):
        for name, config_fn in self.module_lookup.items():
            self.modules[name].update(config_fn())


    @abstractmethod
    def shape_in(self) -> list[int]:
        pass

    @abstractmethod
    def shape_out(self) -> list[int]:
        pass

    @abstractmethod
    def rate_in(self) -> float:
        pass

    @abstractmethod
    def rate_out(self) -> float:
        pass

    def streams_in(self) -> int:
            """
            Returns the number of input streams for this layer.

            Returns:
                int: The number of input streams for this layer.
            """
            return self.coarse_in

    def streams_out(self) -> int:
            """
            Returns the number of output streams for this layer.

            Returns:
                int: The number of output streams for this layer.
            """
            return self.coarse_out

    @abstractmethod
    def width_in(self):
        pass

    @abstractmethod
    def width_out(self):
        pass

    def workload_in(self) -> int:
        """
        Returns the total number of elements in the input tensor of this layer.

        Returns:
            int: The total number of elements in the input tensor of this layer.
        """
        return np.prod(self.shape_in())

    def workload_out(self) -> int:
            """
            Calculates the total number of output elements for this layer.

            Returns:
                The total number of output elements for this layer.
            """
            return np.prod(self.shape_out())

    def size_in(self) -> int:
        return self.workload_in() / self.streams_in()

    def size_out(self) -> int:
        return self.workload_out() / self.streams_out()

    def latency_in(self):
        """
        Calculates the latency of input data for this layer.

        Returns:
            int: The calculated latency value.
        """
        return int(abs(self.workload_in()/(min(self.mem_bw_in, self.rate_in()*self.streams_in()))))

    def latency_out(self):
        """
        Calculates the latency of the output of the layer.

        Returns:
            int: The latency of the output of the layer.
        """
        return int(abs(self.workload_out()/(min(self.mem_bw_out, self.rate_out()*self.streams_out()))))

    def latency(self):
        # return max(self.latency_in(), self.latency_out())
        return max([ self.modules[module].latency() for module in self.modules ])

    def pipeline_depth(self):
            """
            Computes the pipeline depth of the layer by summing the pipeline depths of its modules.

            Returns:
                The pipeline depth of the layer.
            """
            return sum([ self.modules[module].pipeline_depth() for module in self.modules ])

    # def wait_depth(self):
    #     return sum([ self.modules[module].wait_depth() for module in self.modules ])

    @abstractmethod
    def resource(self):
        pass

    def memory_bandwidth(self):
        return {
            "in"  : min(self.mem_bw_in, self.rate_in()*self.streams_in()),
            "out" : min(self.mem_bw_out, self.rate_out()*self.streams_out())
        }

    @abstractmethod
    def get_coarse_in_feasible(self):
        pass

    @abstractmethod
    def get_coarse_out_feasible(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def layer_info(self, parameters, batch_size=1):
            """
            Populates a `Parameters` object with information about this layer.

            Args:
                parameters (Parameters): The `Parameters` object to populate.
                batch_size (int, optional): The batch size to use for the `Parameters` object. Defaults to 1.
            """
            parameters.batch_size   = batch_size
            parameters.coarse_in    = self.coarse_in
            parameters.coarse_out   = self.coarse_out
            parameters.mem_bw_in    = self.mem_bw_in
            parameters.mem_bw_out   = self.mem_bw_out
            self.data_t.to_protobuf(parameters.data_t)

    def get_operations(self):
        return 0

    def get_sparse_operations(self):
        return self.get_operations()

    def layer_info_dict(self):
        # get parameters
        parameter = fpgaconvnet_pb2.parameter()
        self.layer_info(parameter)
        # convert to dictionary
        return MessageToDict(parameter, preserving_proto_field_name=True)

    def visualise(self,name):
        raise NotImplementedError

    @abstractmethod
    def functional_model(self, data, batch_size=1):
        raise NotImplementedError(f"Functional model not implemented for layer type: {self.__class__.__name__}")

@dataclass(kw_only=True)
class Layer2D(LayerBase):
    rows: int
    cols: int
    channels: int

    dimensionality: ClassVar[DIMENSIONALITY] = DIMENSIONALITY.TWO

    def __post_init__(self):
        super().__post_init__()

    def rows_in(self) -> int:
        return self.rows

    def cols_in(self) -> int:
       return self.cols

    def channels_in(self) -> int:
       return self.channels

    def rows_out(self) -> int:
       return self.rows

    def cols_out(self) -> int:
       return self.cols

    def channels_out(self) -> int:
       return self.channels

    def build_rates_graph(self):

        # create the rates graph
        rates_graph = np.zeros(shape=(len(self.modules.keys()),
                                      len(self.modules.keys())+1) , dtype=float )

        # iterate over modules
        for i, module in enumerate(self.modules.keys()):
            # update rates_graph
            rates_graph[i,i] = self.modules[module].rate_in[0]
            rates_graph[i,i+1] = self.modules[module].rate_out[0]

        # return rates_graph
        return rates_graph

    def rate_in(self) -> float:
        return abs(balance_module_rates(self.build_rates_graph())[0,0])

    def rate_out(self) -> float:
        return abs(balance_module_rates(
            self.build_rates_graph())[len(self.modules.keys())-1,len(self.modules.keys())])

    def shape_in(self) -> list[int]: # TODO: add documentation
        return [ self.rows_in(), self.cols_in(), self.channels_in() ]

    def shape_out(self) -> list[int]: # TODO: add documentation
        return [ self.rows_out(), self.cols_out(), self.channels_out() ]

    def width_in(self):
        return self.data_t.width

    def width_out(self):
       return self.data_t.width

    def get_coarse_in_feasible(self):
        return get_factors(int(self.channels_in()))

    def get_coarse_out_feasible(self):
        return get_factors(int(self.channels_out()))

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(self, parameters, batch_size)
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()

    # def visualise(self,name):
    #     cluster = pydot.Cluster(name,label=name)

    #     for i in range(self.coarse_in):
    #         cluster.add_node(pydot.Node( "_".join([name,"edge",str(i)]), label=self.__class__.__name__ ))

    #     return cluster, "_".join([name,"edge"]), "_".join([name,"edge"])

@dataclass(kw_only=True)
class Layer3D(Layer2D):
    depth: int

    dimensionality: ClassVar[DIMENSIONALITY] = DIMENSIONALITY.THREE

    def depth_in(self) -> int:
        return self.depth

    def depth_out(self) -> int:
       return self.depth

    def shape_in(self) -> list[int]: # TODO: add documentation
        return [ self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in() ]

    def shape_out(self) -> list[int]: # TODO: add documentation
        return [ self.rows_out(), self.cols_out(), self.depth_out(), self.channels_out() ]

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(self, parameters, batch_size)
        parameters.depth_in     = self.depth_in()
        parameters.depth_out    = self.depth_out()

