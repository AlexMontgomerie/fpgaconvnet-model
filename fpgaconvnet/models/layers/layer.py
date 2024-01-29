"""

"""
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Optional, List
import math

import networkx as nx # type: ignore
import numpy as np
import pydot # type: ignore
from dacite import from_dict
from google.protobuf.json_format import MessageToDict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.layers.utils import balance_module_rates, get_factors
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model
from fpgaconvnet.models.exceptions import LayerNotImplementedError, AmbiguousLayerError

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
    def get_all_layers(cls, name: str, backend: BACKEND, dimensionality: DIMENSIONALITY):

        # get all the modules in the registry
        layers = list(cls.LAYER_REGISTRY.values())

        # filter all the modules with the given name
        layers = list(filter(lambda m: m.name == name, layers))

        # filter all the modules with the given backend
        layers = list(filter(lambda m: m.backend == backend, layers))

        # filter all the modules with the given dimensionality
        layers = list(filter(lambda m: dimensionality == m.dimensionality, layers))

        return layers

    @classmethod
    def build(cls, name: str, config: dict, backend: BACKEND, dimensionality: DIMENSIONALITY):

        # get all the relevant layers
        layers = cls.get_all_layers(name, backend, dimensionality)

        # check there is at least 1 module
        if len(layers) == 0:
            raise LayerNotImplementedError(f"No layers found for name={name}, \
                    backend={backend.name}, dimensionality={dimensionality.value}")

        # check there is only a single module left
        if len(layers) > 1:
            raise AmbiguousLayerError(f"Too many layers found for name={name}, \
                    backend={backend.name}, dimensionality={dimensionality.value}")

        # get the module class
        layer= layers[0]

        # create a new instance of the module
        return from_dict(data_class=layer, data=config)
        # inst.__post_init__()
        # return inst


    @classmethod
    def build_from_dict(cls, name: str, conf: dict):
        return from_dict(data_class=cls.LAYER_REGISTRY[name], data=conf) # type: ignore
        # inst.__post_init__()
        # return inst

@dataclass(kw_only=True)
class LayerBase(metaclass=LayerBaseMeta):

    coarse_in: int = 1
    coarse_out: int = 1
    mem_bw_in: float = 100.0
    mem_bw_out: float = 100.0
    data_t: FixedPoint = FixedPoint(16,8)
    input_compression_ratio: List[float] = field(default_factory=lambda: [1.0], init=True)
    output_compression_ratio: List[float] = field(default_factory=lambda: [1.0], init=True)
    modules: dict = field(default_factory=OrderedDict, init=True)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph, init=True)

    name: ClassVar[str]
    backend: ClassVar[BACKEND]
    dimensionality: ClassVar[DIMENSIONALITY]
    register: ClassVar[bool] = False

    def __post_init__(self):

        self.input_t = self.data_t
        self.output_t = self.data_t
        self.stream_inputs = [False]
        self.stream_outputs = [False]

        # create the module instances
        for name, config_fn in self.module_lookup.items():
            self.modules[name] = ModuleBase.build(
                    name, config_fn(), self.backend, self.dimensionality)

        # create the module graph
        self.build_module_graph()

        # set as initialised
        self.is_init = True

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

        try:
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

        except AttributeError:
            print(f"WARNING: unable to set attribute {name}, trying super method")
            super().__setattr__(name, value)


    def update(self):
        for name, config_fn in self.module_lookup.items():
            self.modules[name].update(config_fn())

    @abstractmethod
    def build_module_graph(self) -> nx.DiGraph: ...

    @abstractmethod
    def input_shape(self, port_idx: int = 0) -> list[int]: ...

    @abstractmethod
    def output_shape(self, port_idx: int = 0) -> list[int]: ...

    @abstractmethod
    def input_shape_dict(self, port_idx: int = 0) -> dict[str,int]: ...

    @abstractmethod
    def output_shape_dict(self, port_idx: int = 0) -> dict[str,int]: ...

    def rate_in(self, port_idx: int = 0) -> float:
        return self.size_in(port_idx) / float(self.latency())

    def rate_out(self, port_idx: int = 0) -> float:
        return self.size_out(port_idx) / float(self.latency())

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

    # @abstractmethod
    # def width_in(self):
    #     pass

    # @abstractmethod
    # def width_out(self):
    #     pass

    def workload_in(self, port_idx: int = 0) -> int:
        """
        Returns the total number of elements in the input tensor of this layer.

        Returns:
            int: The total number of elements in the input tensor of this layer.
        """
        return math.prod(self.input_shape(port_idx))

    def workload_out(self, port_idx: int = 0) -> int:
        """
        Calculates the total number of output elements for this layer.

        Returns:
            The total number of output elements for this layer.
        """
        return math.prod(self.output_shape(port_idx))

    def size_in(self, port_idx: int = 0) -> int:
        return self.workload_in(port_idx) // self.streams_in()

    def size_out(self, port_idx: int = 0) -> int:
        return self.workload_out(port_idx) // self.streams_out()

    def latency(self) -> int:
        # return max(self.latency_in(), self.latency_out())
        return max([ module.latency() for module in self.modules.values() ])

    def pipeline_depth(self) -> int:
            """
            Computes the pipeline depth of the layer by summing the pipeline depths of its modules.

            Returns:
                The pipeline depth of the layer.
            """
            return sum([ self.modules[module].pipeline_depth() for module in self.modules ])

    @abstractmethod
    def start_depth(self) -> int: ...

    # def wait_depth(self):
    #     return sum([ self.modules[module].wait_depth() for module in self.modules ])

    def resource(self, model: Optional[ResourceModel] = None) -> dict[str,int]:

        # initialise resources
        resources = {
            "DSP": 0,
            "LUT": 0,
            "FF": 0,
            "BRAM": 0,
        }

        # iterate over the nodes of the graph
        for node in self.graph.nodes:

            # get the module
            module = self.graph.nodes[node]["module"]

            # iter over resource types
            for rsc_type in resources.keys():

                # evaluate the resource model
                resources[rsc_type] += eval_resource_model(module, rsc_type, _model=model)

        # return the resources
        return resources

    def memory_bandwidth(self):
        return {
            "in"  : min(self.mem_bw_in, self.rate_in()*self.streams_in()),
            "out" : min(self.mem_bw_out, self.rate_out()*self.streams_out())
        }

    @abstractmethod
    def get_coarse_in_feasible(self): ...

    @abstractmethod
    def get_coarse_out_feasible(self): ...

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
    def functional_model(self, data, batch_size=1): ...
        # raise NotImplementedError(f"Functional model not implemented for layer type: {self.__class__.__name__}")


