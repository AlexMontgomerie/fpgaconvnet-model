'''
Base class for all hardware module models.
'''

import re
import importlib
import numpy as np
import math
import os
import copy
from typing import List
from dataclasses import dataclass, field, asdict
from xgboost import XGBRegressor
from abc import ABC, ABCMeta, abstractmethod
from typing import ClassVar, Dict
from dacite import from_dict

from fpgaconvnet.data_types import FixedPoint

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
        if new_cls.__name__ != "ModuleBase":
            cls.MODULE_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.MODULE_REGISTRY)

    @classmethod
    def get_all_modules(cls, name: str, backend: str, dimensionality: int):

        # get all the modules in the registry
        modules = list(cls.MODULE_REGISTRY.values())

        # filter all the modules with the given name
        modules = list(filter(lambda m: m.name == name, modules))

        # filter all the modules with the given backend
        modules = list(filter(lambda m: m.backend == backend, modules))

        # filter all the modules with the given dimensionality
        modules = list(filter(lambda m: m.dimensionality == dimensionality, modules))

        return modules

    @classmethod
    def build(cls, name: str, config: dict, backend: str, dimensionality: int):

        # get all the relevant modules
        modules = cls.get_all_modules(name, backend, dimensionality)

        # check there is only a single module left
        assert(len(modules) == 1, f"Too many modules found for name={name}, \
                backend={backend}, dimensionality={dimensionality}")

        # get the module class
        module = modules[0]

        # create a new instance of the module
        return from_dict(data_class=module, data=config)

@dataclass(kw_only=True)
class ModuleBase(metaclass=ModuleBaseMeta):
    name: ClassVar[str]
    backend: ClassVar[str]
    dimensionality: ClassVar[int]

    ports_in: ClassVar[int]
    ports_out: ClassVar[int]

    repetitions: int = 1

    @property
    @abstractmethod
    def input_ports(self) -> List[Port]:
        pass

    @property
    @abstractmethod
    def output_ports(self) -> List[Port]:
        pass

    @abstractmethod
    def functional_model(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def rate_in(self, idx: int) -> float:
        pass

    @abstractmethod
    def rate_out(self, idx: int) -> float:
        pass

    @abstractmethod
    def resource_parameters(self) -> list[int]:
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


@dataclass
class Module:
    """
    modules are the fundamental building block for the hardware
    framework. In this base class, performance and resource model
    templates are included, as well as a template for functional
    models. All modules are derived from this base class and contain
    the same methods.

    Attributes
    ----------
    rows: int
        row dimension of input featuremap
    cols: int
        column dimension of input featuremap
    channels: int
        channel dimension of input featuremap
    data_width: int
        bitwidth of featuremap pixels
    rsc_coef: list
        list of resource model coefficients. Corresponds
        to `LUT`, `BRAM`, `DSP` and `FF` resources in
        that order.

    .. note::
        The model expects that the module is run for a single three
        dimensional featuremap. For intermediate modules within a layer,
        they may not be operating on a three dimensional tensor, and
        so the `rows`, `cols` and `channels` attributes are representative
        of the tensor if it was flattened to three dimensions.
    """

    rows: int
    cols: int
    channels: int
    data_width: int = field(default=16, init=False)
    rsc_coef: dict = field(default_factory=lambda: {
        "FF": [], "LUT": [], "DSP": [], "BRAM": []}, init=False)
    streams: int = field(default=1, init=False)
    backend: str = field(default="chisel", init=False)
    regression_model: str = field(default="linear_regression", init=False)
    latency_mode: bool = field(default=False, init=False)
    block: bool = field(default=False, init=False)

    def __post_init__(self):

        # get the module identifer
        self.module_identifier = self.__class__.__name__

        # load resource coefficients
        self.load_resource_coefficients(self.module_identifier)

    def load_resource_coefficients(self, module_identifier):

        # get the cache path
        rsc_cache_path = os.path.dirname(__file__) + \
            f"/../../coefficients/{self.regression_model}/{self.backend}"

        # iterate over resource types
        self.rsc_coef = {}
        self.rsc_model = {}
        for rsc_type in self.utilisation_model():
            match self.regression_model:
                case "linear_regression":
                    # get the coefficients from the cache path and load
                    coef_path = os.path.join(rsc_cache_path,
                            f"{module_identifier}_{rsc_type}.npy".lower())
                    self.rsc_coef[rsc_type] = np.load(coef_path)
                case "xgboost" | "xgboost-kernel":
                    # get the coefficients from the cache path and load
                    model_path = os.path.join(rsc_cache_path,
                            f"{module_identifier}_{rsc_type}.json".lower())
                    model = XGBRegressor(n_jobs=4)
                    model.load_model(model_path)
                    self.rsc_model[rsc_type] = model
                case _:
                    raise NotImplementedError(f"{self.regression_model} not supported")

    def utilisation_model(self):
        raise ValueError(f"{self.backend} backend not supported")

    def get_pred_array(self):
        raise ValueError(f"{self.backend} backend not supported")

    def rsc(self, coef=None, model=None):
        """
        Returns
        -------
        dict
            estimated resource usage of the module. Uses the
            resource coefficients for the estimate.
        """

        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        if model == None:
            model = self.rsc_model

        # return the linear model estimation
        match self.regression_model:
            case "linear_regression":
                # get the utilisation_model
                util_model = self.utilisation_model()
                rsc = { rsc_type: int(np.dot(util_model[rsc_type],
                    coef[rsc_type])) for rsc_type in coef.keys()}
            case "xgboost-kernel":
                util_model = self.utilisation_model()
                rsc = { rsc_type: int(model[rsc_type].predict(
                    np.expand_dims(util_model[rsc_type], axis=0))) for rsc_type in model.keys()}
            case "xgboost":
                pred_array = self.get_pred_array()
                rsc = { rsc_type: int(model[rsc_type].predict(pred_array)) for rsc_type in model.keys()}
            case _:
                raise NotImplementedError(f"{self.regression_model} not supported")

        if self.backend == "chisel":
            # update the resources for sum of LUT and BRAM types
            rsc["LUT"] = rsc["Logic_LUT"] + rsc["LUT_RAM"] + rsc["LUT_SR"]
            rsc["BRAM"] = 2*rsc["BRAM36"] + rsc["BRAM18"]

        # return updated resources
        return rsc

    def module_info(self):
        """
        creates a dictionary containing information and
        parameters for the module.
        """
        return {
            'type'      : self.__class__.__name__.upper(),
            'streams'       : self.streams,
            'rows_in'       : self.rows_in(),
            'cols_in'       : self.cols_in(),
            'channels_in'   : self.channels_in(),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def rows_in(self):
        """
        Returns
        -------
        int
            row dimension of the input featuremap
        """
        return self.rows

    def cols_in(self):
        """
        Returns
        -------
        int
            column dimension of the input featuremap
        """
        return self.cols

    def channels_in(self):
        """
        Returns
        -------
        int
            channel dimension of the input featuremap
        """
        return self.channels

    def rows_out(self):
        """
        Returns
        -------
        int
            row dimension of the output featuremap
        """
        return self.rows

    def cols_out(self):
        """
        Returns
        -------
        int
            column dimension of the output featuremap
        """
        return self.cols

    def channels_out(self):
        """
        Returns
        -------
        int
            channel dimension of the output featuremap
        """
        return self.channels

    def rate_in(self):
        """
        Returns
        -------
        float
            rate of words into module. As a fraction of a
            clock cycle.

            default is 1.0
        """
        return 1.0

    def rate_out(self):
        """
        Returns
        -------
        float
            rate of words out of the module. As a fraction
            of a clock cycle.

            default is 1.0
        """
        return 1.0

    def latency(self):
        """
        Returns
        -------
        int
            calculates the number of clock cycles latency
            it takes for the module to process a featuremap.
            First latency in and latency out is calculated,
            then the latency of the module is the largest of
            the two.
        """
        latency_in  = int((self.rows_in() *self.cols_in() *self.channels_in() )/self.rate_in() )
        latency_out = int((self.rows_out()*self.cols_out()*self.channels_out())/self.rate_out())
        return max(latency_in,latency_out)

    def pipeline_depth(self):
        """
        Returns
        -------
        int
           depth of the pipeline for the module in clock
           cycles.

           default is 0.
        """
        return 0

    def memory_usage(self):
        """
        Returns
        -------
        int
            number of memory bits required by the module.

            default is 0.
        """
        return 0


    def functional_model(self,data):
        """
        functional model of the module. Used for verification
        of hardware modules.

        Returns
        -------
        np.array
        """
        return data

