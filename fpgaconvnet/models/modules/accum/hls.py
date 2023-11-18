from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import random

from fpgaconvnet.models.modules import Port, ModuleBaseMeta, ModuleHLSBase, ModuleHLS3DBase, int2bits
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.data_types import FixedPoint

@dataclass
class AccumHLSBase(ModuleHLSBase):

    # hardware parameters
    filters: int
    groups: int
    accum_t: FixedPoint = FixedPoint(32, 16)

    # class variables
    name: ClassVar[str] = "accum"
    register: ClassVar[bool] = False

    @classmethod
    def generate_random_configuration(cls):

        # get the HLS base configuration
        config = super().generate_random_configuration()

        # generate a random configuration
        data_width = random.choice([4, 8, 16, 32])
        config.update({
            "filters": random.randint(1, 512),
            "groups": 1, # TODO: explore different values
            "accum_t": {
                "width": data_width,
                "binary_point": random.randint(0, data_width-1)
            },
            "activity": {
                "input": random.uniform(0, 1.0),
            },
            "description": "Randomly generated Accum configuration"
        })

        # return the configuration
        return config

    def rate_in(self, idx: int) -> float:
        super().rate_in(idx)
        return 1.0

    def rate_out(self, idx: int) -> float:
        super().rate_out(idx)
        return self.groups/float(self.channels)

    def pipeline_depth(self):
        # return (self.channels*self.filters)//(self.groups*self.groups)
        return self.channels//self.groups

    def functional_model(self, data: np.ndarray) -> np.ndarray:

        # check input dimensionality
        iter_space_len = len(self.input_iteration_space(0))
        assert len(data.shape) >= iter_space_len
        assert data.shape[-iter_space_len] == self.input_iteration_space(0)

        # accumulate across the channel dimension
        return np.sum(data, axis=-3)

@dataclass
class AccumHLS(AccumHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            iteration_space=[self.rows, self.cols, self.channels, self.filters//self.groups],
            simd_lanes=[1],
            data_type=self.accum_t,
            buffer_depth=0,
            name="in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            iteration_space=[self.rows, self.cols, self.filters//self.groups],
            simd_lanes=[1],
            data_type=self.accum_t,
            buffer_depth=0,
            name="out"
        )]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.groups, self.channels, self.filters, self.accum_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"   : np.array([
                self.filters, self.groups, self.accum_t.width,
                self.cols, self.rows, self.channels
            ]),
            "FF"    : np.array([
                self.filters, self.groups, self.accum_t.width,
                self.cols,self.rows,self.channels
            ]),
            "DSP"   : np.array([
                self.filters, self.groups ,self.accum_t.width,
                self.cols, self.rows, self.channels
            ]),
            "BRAM"  : np.array([
                self.filters, self.groups, self.accum_t.width,
                self.cols, self.rows, self.channels
            ]),
        }

@dataclass
class AccumHLS3D(ModuleHLS3DBase, AccumHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            iteration_space=[self.rows, self.cols, self.depth, self.channels, self.filters//self.groups],
            simd_lanes=[1],
            data_type=self.accum_t,
            buffer_depth=0,
            name="in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            iteration_space=[self.rows, self.cols, self.depth, self.filters//self.groups],
            simd_lanes=[1],
            data_type=self.accum_t,
            buffer_depth=0,
            name="out"
        )]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.groups, self.channels, self.filters, self.accum_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"   : np.array([
                self.filters, self.groups, self.accum_t.width,
                self.cols, self.rows, self.channels, self.depth
            ]),
            "FF"    : np.array([
                self.filters, self.groups, self.accum_t.width,
                self.cols, self.rows, self.channels, self.depth
            ]),
            "DSP"   : np.array([
                self.filters, self.groups ,self.accum_t.width,
                self.cols, self.rows, self.channels, self.depth
            ]),
            "BRAM"  : np.array([
                self.filters, self.groups, self.accum_t.width,
                self.cols, self.rows, self.channels, self.depth
            ]),
        }

