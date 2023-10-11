'''
Base class for all hardware module models.
'''

import re
import importlib
import math
import os
import copy
from typing import List
from dataclasses import dataclass, field

import numpy as np
from xgboost import XGBRegressor
from dacite import from_dict

from fpgaconvnet.data_types import FixedPoint

# from fpgaconvnet.tools.resource_regression_model import ModuleModel

RSC_TYPES = ["FF","LUT","DSP","BRAM"]

SERVER_DB="mongodb+srv://fpgaconvnet.hwnxpyo.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"

class ModuleMeta(type):

    MODULE_REGISTRY = {}

    def __new__(cls, *args, **kwargs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = super().__new__(cls, *args, **kwargs)
        cls.MODULE_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.MODULE_REGISTRY)

    @classmethod
    def build_from_dict(cls, name: str, conf: dict):
        inst = from_dict(data_class=cls.MODULE_REGISTRY[name], data=conf)
        inst.__post_init__()
        return inst


@dataclass(kw_only=True)
class Module(metaclass=ModuleMeta):
    repeat: int = 1 # the number of times the execution of the input shape is repeated
    streams: int = 1 # number of parallel streams into the module
    blocks: int = 1 # number of parallel hardware blocks to execute the streams
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    data_t: int = field(default_factory=lambda: FixedPoint(16, 8), init=True)
    rsc_coef: dict = field(default_factory=lambda: {
        "FF": [], "LUT": [], "DSP": [], "BRAM": []}, init=False)

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

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

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
            'type'          : self.__class__.__name__.upper(),
            'repeat'        : self.repeat,
            'streams'       : self.streams,
            'blocks'        : self.blocks,
            'input_shape'   : self.input_shape,
            'output_shape'  : self.output_shape,
            'data_t'        : self.data_t.to_dict(),
        }

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
        latency_in  = self.repeat*int(np.prod(self.input_shape)/self.rate_in())
        latency_out = self.repeat*int(np.prod(self.output_shape)/self.rate_in())
        return max(latency_in, latency_out)

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
        raise NotImplementedError
