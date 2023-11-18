"""
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model

@dataclass
class EltWise(Module):
    ports_in: int
    eltwise_type: str
    broadcast: bool = False
    biases_width: int = field(default=16, init=False)
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):
        pass

    def rsc(self, coef=None, model=None):
        """
        Returns
        -------
        dict
            estimated resource usage of the module. Uses the
            resource coefficients for the estimate.
        """
        # get the channel buffer BRAM estimate
        channel_buffer_bram = bram_array_resource_model(int(self.channels), self.data_width, "fifo")

        return {
            "LUT"   : 49,
            "FF"    : 23,
            "BRAM"  : channel_buffer_bram if self.broadcast else 0,
            "DSP"   : 0 if self.eltwise_type == "add" else 1
        }

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'ports_in'      : self.ports_in,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def utilisation_model(self):

        if self.backend == "hls":
            return {
                "LUT"   : np.array([1]),
                "FF"    : np.array([1]),
                "DSP"   : np.array([0]),
                "BRAM"  : np.array([0]),
            }

        if self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([1]),
                "LUT_RAM"   : np.array([1]),
                "LUT_SR"    : np.array([1]),
                "FF"        : np.array([1]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }

        else:
            raise ValueError(f"{self.backend} backend not supported")

    def functional_model(self, data):
        # check input dimensionality
        assert len(data) == self.ports_in , "ERROR: invalid row dimension"
        for i in range(self.ports_in):
            assert data[i].shape[0] == self.rows        , "ERROR: invalid column dimension"
            assert data[i].shape[1] == self.cols        , "ERROR: invalid column dimension"
            assert data[i].shape[2] == self.channels    , "ERROR: invalid channel dimension"

        if self.eltwise_type == "add":
            out = np.zeros((
                self.rows,
                self.cols,
                self.channels),dtype=float)

            for index, _ in np.ndenumerate(out):
                for i in range(self.ports_in):
                    out[index] += float(data[i][index])
        elif self.eltwise_type == "mul":
            out = np.ones((
                self.rows,
                self.cols,
                self.channels),dtype=float)

            for index, _ in np.ndenumerate(out):
                for i in range(self.ports_in):
                    out[index] *= float(data[i][index])
        else:
            raise ValueError(f"Element-wise type {self.eltwise_type} not supported")

        return out

