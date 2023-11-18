'''
Base class for all hardware module models.
'''

import re
import importlib
import numpy as np
import math
import os
import copy
import random
from dataclasses import dataclass, field, asdict
from xgboost import XGBRegressor
from abc import ABC, ABCMeta, abstractmethod
from typing import ClassVar, Dict
from dacite import from_dict

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

@dataclass(kw_only=True)
class Port:
    simd_lanes: list[int]
    data_type: FixedPoint
    buffer_depth: int = 0
    name: str = "port"

    @property
    def port_width(self) -> int:
        return self.data_type.width*np.prod(self.simd_lanes)


class ModuleBaseMeta(type, metaclass=ABCMeta):

    # dictionary lookup for modules
    MODULE_REGISTRY = {}

    def __new__(cls, *args, **kwargs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = super().__new__(cls, *args, **kwargs)
        if new_cls.register:
            cls.MODULE_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.MODULE_REGISTRY)

    @classmethod
    def get_all_modules(cls, name: str, backend: str, dimensionality: int):

        # get all the modules in the registry
        modules = list(cls.MODULE_REGISTRY.values())
        print(modules)

        # filter all the modules with the given name
        modules = list(filter(lambda m: m.name == name, modules))

        # filter all the modules with the given backend
        modules = list(filter(lambda m: m.backend == backend, modules))

        # filter all the modules with the given dimensionality
        modules = list(filter(lambda m: dimensionality in m.dimensionality, modules))

        return modules

    @classmethod
    def build(cls, name: str, config: dict, backend: str, dimensionality: int):

        # get all the relevant modules
        modules = cls.get_all_modules(name, backend, dimensionality)

        # check there is at least 1 module
        assert len(modules) > 0, f"No modules found for name={name}, \
                backend={backend}, dimensionality={dimensionality}"

        # check there is only a single module left
        assert len(modules) == 1, f"Too many modules found for name={name}, \
                backend={backend}, dimensionality={dimensionality}"

        # get the module class
        module = modules[0]

        # create a new instance of the module
        return from_dict(data_class=module, data=config)

    @classmethod
    def from_config(cls, config: dict):
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

    @property
    @abstractmethod
    def input_ports(self) -> list[Port]:
        pass

    @property
    @abstractmethod
    def output_ports(self) -> list[Port]:
        pass

    @property
    @abstractmethod
    def input_iter_space(self) -> list[list[int]]:
        pass

    @property
    @abstractmethod
    def output_iter_space(self) -> list[list[int]]:
        pass

    @property
    @abstractmethod
    def rate_in(self) -> list[float]:
        pass

    @property
    @abstractmethod
    def rate_out(self) -> list[float]:
        pass

    @property
    def ports_in(self) -> int:
        return len(self.input_ports)

    @property
    def ports_out(self) -> int:
        return len(self.output_ports)

    @property
    def input_simd_lanes(self) -> list[list[int]]:
        return [ p.simd_lanes for p in self.input_ports ]

    @property
    def output_simd_lanes(self) -> list[list[int]]:
        return [ p.simd_lanes for p in self.output_ports ]

    @abstractmethod
    def functional_model(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def resource_parameters(self) -> list[int]:
        pass

    @abstractmethod
    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        pass

    @abstractmethod
    def pipeline_depth(self) -> int:
        pass

    def get_rate_in(self, idx: int) -> float:
        assert idx < self.ports_in, "Invalid input port index"
        return self.rate_in[idx]

    def get_rate_out(self, idx: int) -> float:
        assert idx < self.ports_out, "Invalid output port index"
        return self.rate_out[idx]

    def get_input_simd_lanes(self, idx: int) -> list[int]:
        assert idx < self.ports_in, "Invalid input port index"
        return self.input_ports[idx].simd_lanes

    def get_output_simd_lanes(self, idx: int) -> list[int]:
        assert idx < self.ports_out, "Invalid output port index"
        return self.output_ports[idx].simd_lanes

    def latency(self) -> int:
        latency_in = max([int(np.prod(self.input_iter_space[i]) \
                / self.rate_in[i]) for i in range(self.ports_in)])
        latency_out = max([int(np.prod(self.output_iter_space[i]) \
                / self.rate_out[i]) for i in range(self.ports_out)])
        return max(latency_in, latency_out)

    def module_info(self) -> dict:
        return asdict(self)

    def resource_parameters_heuristics(self) -> Dict[str, int]:
        return {}

@dataclass(kw_only=True)
class ModuleChiselBase(ModuleBase):

    streams: int = 1
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    backend: ClassVar[BACKEND] = BACKEND.CHISEL
    dimensionality: ClassVar[set[DIMENSIONALITY]] = { DIMENSIONALITY.TWO, DIMENSIONALITY.THREE }
    register: ClassVar[bool] = False

@dataclass(kw_only=True)
class ModuleHLSBase(ModuleBase):

    rows: int
    cols: int
    channels: int
    batch_size: int = 1

    backend: ClassVar[BACKEND] = BACKEND.HLS
    dimensionality: ClassVar[set[DIMENSIONALITY]] = { DIMENSIONALITY.TWO }
    register: ClassVar[bool] = False

@dataclass(kw_only=True)
class ModuleHLS3DBase(ModuleHLSBase):
    depth: int

    dimensionality: ClassVar[set[DIMENSIONALITY]] = { DIMENSIONALITY.THREE }
    register: ClassVar[bool] = False

class Module:
    pass
