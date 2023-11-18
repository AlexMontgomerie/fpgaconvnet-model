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
from typing import List
from dataclasses import dataclass, field, asdict
from xgboost import XGBRegressor
from abc import ABC, ABCMeta, abstractmethod
from typing import ClassVar, Dict
from dacite import from_dict

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

@dataclass(kw_only=True)
class Port:
    iteration_space: list[int]
    simd_lanes: list[int]
    data_type: FixedPoint
    buffer_depth: int = 0
    name: str = "port"

    @property
    def samples(self) -> int:
        return np.prod(self.iteration_space)//np.prod(self.simd_lanes)

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

    @classmethod
    @abstractmethod
    def generate_random_configuration(cls):
        pass

    @classmethod
    def build_random_inst(cls):

        # generate a random configuration
        config = cls.generate_random_configuration()

        # create a new instance of the module
        return from_dict(data_class=cls, data=config)


@dataclass(kw_only=True)
class ModuleBase(metaclass=ModuleBaseMeta):
    name: ClassVar[str]
    backend: ClassVar[BACKEND]
    dimensionality: ClassVar[set[DIMENSIONALITY]]
    register: ClassVar[bool] = False
    repetitions: int = 1

    @property
    @abstractmethod
    def input_ports(self) -> List[Port]:
        pass

    @property
    @abstractmethod
    def output_ports(self) -> List[Port]:
        pass

    @property
    def ports_in(self) -> int:
        return len(self.input_ports)

    @property
    def ports_out(self) -> int:
        return len(self.output_ports)

    @abstractmethod
    def functional_model(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def rate_in(self, idx: int) -> float:
        assert idx < self.ports_in, "Invalid input port index"

    @abstractmethod
    def rate_out(self, idx: int) -> float:
        assert idx < self.ports_out, "Invalid output port index"

    @abstractmethod
    def resource_parameters(self) -> list[int]:
        pass

    @abstractmethod
    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        pass

    @abstractmethod
    def pipeline_depth(self) -> int:
        pass

    def input_iter_space(self, idx: int) -> List[int]:
        return self.input_ports[idx].iteration_space

    def output_iter_space(self, idx: int) -> List[int]:
        return self.output_ports[idx].iteration_space

    def input_simd_lanes(self, idx: int) -> List[int]:
        return self.input_ports[idx].simd_lanes

    def output_simd_lanes(self, idx: int) -> List[int]:
        return self.output_ports[idx].simd_lanes

    def latency(self) -> int:
        latency_in = self.repetitions * max([int(self.input_ports[i].samples \
                / self.rate_in(i)) for i in range(self.ports_in)])
        latency_out = self.repetitions * max([int(self.output_ports[i].samples \
                / self.rate_out(i)) for i in range(self.ports_out)])
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

    @classmethod
    def generate_random_configuration(cls):

        # generate a random configuration
        config = {
            "rows": random.randint(1, 128),
            "cols": random.randint(1, 128),
            "channels": random.randint(1, 512),
        }

        # return the configuration
        return config

@dataclass(kw_only=True)
class ModuleHLS3DBase(ModuleHLSBase):
    depth: int

    dimensionality: ClassVar[set[DIMENSIONALITY]] = { DIMENSIONALITY.THREE }
    register: ClassVar[bool] = False

    @classmethod
    def generate_random_configuration(cls):

        # get the 2D parameters
        config = super().generate_random_configuration()

        # add the 3D parameters
        config["depth"] = random.randint(1, 128)

        # return the configuration
        return config


class Module:
    pass
