'''
Base class for all hardware module models.
'''

import numpy as np
from dataclasses import dataclass, asdict
from xgboost import XGBRegressor
from abc import ABCMeta, abstractmethod
from typing import ClassVar
from dacite import from_dict

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.architecture import Architecture, BACKEND, DIMENSIONALITY
from fpgaconvnet.models.exceptions import ModuleNotImplementedError, AmbiguousModuleError

@dataclass(kw_only=True)
class Port:
    """
    A port object which describes the interface of a module.
    """
    simd_lanes: list[int]
    """The number of synchronised parallel words in the port."""
    data_type: FixedPoint
    """The data type of singular words in the SIMD lane."""
    buffer_depth: int = 0
    """The depth of the elastic buffer."""
    name: str = "port"
    """The name of the port, typically either "in" or "out"."""

    @property
    def port_width(self) -> int:
        """
        The total width of the SIMD lane, in terms of bits.
        """
        return self.data_type.width*int(np.prod(self.simd_lanes))


class ModuleBaseMeta(type, metaclass=ABCMeta):

    name: ClassVar[str]
    backend: ClassVar[BACKEND]
    dimensionality: ClassVar[set[DIMENSIONALITY]]
    register: ClassVar[bool] = False

    # dictionary lookup for modules
    MODULE_REGISTRY: dict[str, object] = {}

    def __new__(cls, *args, **kwargs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = super().__new__(cls, *args, **kwargs)
        if new_cls.register:
            cls.MODULE_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls) -> dict[str, object]:
        """
        A registry containing a lookup of all the modules
        by their class name, and a reference to the class object.

        Returns:
            A dictionary containing the module registry.
        """
        return dict(cls.MODULE_REGISTRY)

    @classmethod
    def get_all_modules(cls, name: str, backend: BACKEND,
                        dimensionality: DIMENSIONALITY) -> list[object]:
        """
        Get all the modules in the registry with the given name,
        backend and dimensionality. Typically there is only a
        single module for a given name, backend and dimensionality.

        Args:
            name: The name of the module.
            backend: The backend of the module.
            dimensionality: The dimensionality of the module.

        Returns:
            A list of module classes.
        """
        # get all the modules in the registry
        modules = list(cls.MODULE_REGISTRY.values())

        # filter all the modules with the given name
        name_filter_fn = lambda m: m.name == name
        modules = list(filter(name_filter_fn, modules))

        # filter all the modules with the given backend
        backend_filter_fn = lambda m: m.backend == backend
        modules = list(filter(backend_filter_fn, modules))

        # filter all the modules with the given dimensionality
        dim_filter_fn = lambda m: dimensionality in m.dimensionality
        modules = list(filter(dim_filter_fn, modules))

        return modules

    @classmethod
    def build(cls, name: str, config: dict, backend: BACKEND,
              dimensionality: DIMENSIONALITY) -> object:
        """
        Build a module from a given name, configuration, backend and dimensionality.

        Args:
            name: The name of the module.
            config: The configuration of the module.
            backend: The backend of the module.
            dimensionality: The dimensionality of the module.

        Returns:
            A new instance of the module.
        """
        # get all the relevant modules
        modules = cls.get_all_modules(name, backend, dimensionality)

        # check there is at least 1 module
        if len(modules) == 0:
            raise ModuleNotImplementedError(f"No modules found for name={name}, \
                    backend={backend.name}, dimensionality={dimensionality.value}")

        # check there is only a single module left
        if len(modules) > 1:
            raise AmbiguousModuleError(f"Too many modules found for name={name}, \
                    backend={backend.name}, dimensionality={dimensionality.value}")

        # get the module class
        module = modules[0]

        # create a new instance of the module
        return from_dict(data_class=module, data=config)

    @classmethod
    def build_from_dict(cls, config: dict):
        """
        Build a module from a given configuration dictionary.

        Args:
            config: The configuration of the module.

        Returns:
            A new instance of the module.
        """
        return from_dict(data_class=cls, data=config)

    # @classmethod
    # @abstractmethod
    # def generate_random_configuration(cls):
    #     pass

    # @classmethod
    # def build_random_inst(cls):

    #     # generate a random configuration
    #     config = cls.generate_random_configuration()

    #     # create a new instance of the module
    #     return from_dict(data_class=cls, data=config)


@dataclass(kw_only=True)
class ModuleBase(metaclass=ModuleBaseMeta):
    name: ClassVar[str]
    backend: ClassVar[BACKEND]
    dimensionality: ClassVar[set[DIMENSIONALITY]]
    register: ClassVar[bool] = False
    repetitions: int = 1

    def __post_init__(self):
        """
        NOTE: Need to keep this method in the base class
        """
        pass

    @property
    def class_name(self) -> str:
        """
        The class name of the module.

        Returns:
            The class name of the module.
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def input_ports(self) -> list[Port]:
        """
        The number of input ports for the module.
        Ports are defined as the number of seperate elastic
        interfaces into/out of the module.

        Returns:
            A list of port objects
        """

    @property
    @abstractmethod
    def output_ports(self) -> list[Port]:
        """
        The number of output ports for the module.
        Ports are defined as the number of seperate elastic
        interfaces into/out of the module.

        Returns:
            A list of port objects
        """

    @property
    @abstractmethod
    def input_iter_space(self) -> list[list[int]]:
        """
        The iteration space of the input ports.
        The iteration space essentially describes the shape
        of the data that is being passed through the module.

        Returns:
            A list of lists, where each list describes the
            iteration space of a single input port.
        """

    @property
    @abstractmethod
    def output_iter_space(self) -> list[list[int]]:
        """
        The iteration space of the output ports.
        The iteration space essentially describes the shape
        of the data that is being passed through the module.

        Returns:
            A list of lists, where each list describes the
            iteration space of a single input port.
        """

    @property
    @abstractmethod
    def rate_in(self) -> list[float]:
        """
        The rate of data coming into the module.
        The rate of data is defined as the number of SIMD words
        per cycle that are being passed through the module.
        The rate is always between 0 and 1, where 1 is the
        maximum rate of data.

        Returns:
            A list of floats, where each float describes the
            rate of data for a single port.
        """

    @property
    @abstractmethod
    def rate_out(self) -> list[float]:
        """
        The rate of data leaving the module.
        The rate of data is defined as the number of SIMD words
        per cycle that are being passed through the module.
        The rate is always between 0 and 1, where 1 is the
        maximum rate of data.

        Returns:
            A list of floats, where each float describes the
            rate of data for a single port.
        """

    @property
    def ports_in(self) -> int:
        """
        The number of input ports for the module.

        Returns:
            The number of ports, as an integer.
        """
        return len(self.input_ports)

    @property
    def ports_out(self) -> int:
        """
        The number of output ports for the module.

        Returns:
            The number of ports, as an integer.
        """
        return len(self.output_ports)

    @property
    def input_simd_lanes(self) -> list[list[int]]:
        """
        The number of SIMD lanes for each input port.

        Returns:
            A list of lists, where each list describes
            the number of SIMD lanes for a single port.
        """
        return [ p.simd_lanes for p in self.input_ports ]

    @property
    def output_simd_lanes(self) -> list[list[int]]:
        """
        The number of SIMD lanes for each output port.

        Returns:
            A list of lists, where each list describes
            the number of SIMD lanes for a single port.
        """
        return [ p.simd_lanes for p in self.output_ports ]

    @abstractmethod
    def functional_model(self, *data: np.ndarray) -> np.ndarray:
        """
        A functional model of the module.
        This model is used to generate test stimuli for the
        module, and to verify the correctness of the module
        against the functional model.
        The functionality of this model should be identical
        to the hardware implementation.

        Args:
            data: The input data to the module.

        Returns:
            The output data of the module.
        """

    @abstractmethod
    def resource_parameters(self) -> list[int]:
        """
        A list of parameters which are used to construct
        a data-driven resource model. These parameters are
        chosen to have importance for the resource usage of
        the module.

        Returns:
            A list of integers, where each integer is a
            resource parameter.
        """

    @abstractmethod
    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        """
        A lookup up table of heuristic resource features
        of the module. These features are typically
        non-linear, and relate to a specific type of
        resource, such as BRAM. For example, the number of
        BRAMs are a non-linear function based on the
        depth of memory needed.

        Returns:
            A dictionary of lists, where each list is a
            heuristic resource feature.
        """

    @abstractmethod
    def pipeline_depth(self) -> int:
        """
        The pipeline depth, in terms of cycles, of the module.

        Returns:
            The pipeline depth, as an integer.
            The default is 0 for all modules.
        """
        return 0

    def get_rate_in(self, idx: int) -> float:
        """
        A helper function to get the rate of a specific input port.

        Args:
            idx: The index of the input port.

        Returns:
            The rate of the input port, as a float.
        """
        assert idx < self.ports_in, "Invalid input port index"
        return self.rate_in[idx]

    def get_rate_out(self, idx: int) -> float:
        """
        A helper function to get the rate of a specific output port.

        Args:
            idx: The index of the output port.

        Returns:
            The rate of the output port, as a float.
        """
        assert idx < self.ports_out, "Invalid output port index"
        return self.rate_out[idx]

    def get_input_simd_lanes(self, idx: int) -> list[int]:
        """
        A helper function to get the number of SIMD lanes for a specific input port.

        Args:
            idx: The index of the input port.

        Returns:
            A list of integers, where each integer is the number of SIMD lanes.
        """
        assert idx < self.ports_in, "Invalid input port index"
        return self.input_ports[idx].simd_lanes

    def get_output_simd_lanes(self, idx: int) -> list[int]:
        """
        A helper function to get the number of SIMD lanes for a specific output port.

        Args:
            idx: The index of the output port.

        Returns:
            A list of integers, where each integer is the number of SIMD lanes.
        """
        assert idx < self.ports_out, "Invalid output port index"
        return self.output_ports[idx].simd_lanes

    def cycles(self) -> int:
        """
        The latency of the module, in terms of cycles.

        Returns:
            The cycles of the module, as an integer.
        """
        cycles_in = max([self.repetitions*int(np.prod(self.input_iter_space[i]) \
                / self.rate_in[i]) for i in range(self.ports_in)])
        cycles_out = max([self.repetitions*int(np.prod(self.output_iter_space[i]) \
                / self.rate_out[i]) for i in range(self.ports_out)])
        return max(cycles_in, cycles_out)

    def module_info(self) -> dict:
        """
        A dictionary containing the information of the module.
        This can be used to construct a new instance of the module.

        Returns:
            A dictionary containing the module information.
        """
        info = asdict(self)
        info["name"] = self.name
        info["backend"] = self.backend.name
        info["dimensionality"] = [ d.value for d in self.dimensionality ]
        return info

    def update(self, config: dict) -> object:
        """
        Update the configuration of the module with a new configuration.

        Args:
            config: The new configuration of the module.

        Returns:
            A new instance of the module with the updated configuration.
        """
        # get the current configuration
        current = self.module_info()

        # update the current configuration with
        # the new configuration
        result = {**current, **config}

        # return a new instance of the module with
        # the updated configuration
        self = from_dict(self.__class__, result)

@dataclass(kw_only=True)
class ModuleChiselBase(ModuleBase):
    """
    Base class for all Chisel hardware module models.
    Chisel hardware modules can be used for both 2D and
    3D layers, as they do not have loop bounds like HLS.
    """
    streams: int = 1
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    backend: ClassVar[BACKEND] = BACKEND.CHISEL
    dimensionality: ClassVar[set[DIMENSIONALITY]] = { DIMENSIONALITY.TWO, DIMENSIONALITY.THREE }
    register: ClassVar[bool] = False

@dataclass(kw_only=True)
class ModuleHLSBase(ModuleBase):
    """
    Base class for all HLS hardware module models.
    The base class is for 2D layers only.
    """
    rows: int
    cols: int
    channels: int
    batch_size: int = 1

    backend: ClassVar[BACKEND] = BACKEND.HLS
    dimensionality: ClassVar[set[DIMENSIONALITY]] = { DIMENSIONALITY.TWO }
    register: ClassVar[bool] = False

@dataclass(kw_only=True)
class ModuleHLS3DBase(ModuleHLSBase):
    """
    Base class for all HLS hardware module models.
    The base class is for 3D layers only.
    """
    depth: int

    dimensionality: ClassVar[set[DIMENSIONALITY]] = { DIMENSIONALITY.THREE }
    register: ClassVar[bool] = False

class Module:
    """
    ..todo.. Deprecated class, remove in future.
    """
    pass
