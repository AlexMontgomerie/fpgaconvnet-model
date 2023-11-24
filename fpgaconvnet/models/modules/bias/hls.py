import random
from typing import ClassVar, List
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleHLSBase, ModuleHLS3DBase, Port

@dataclass(kw_only=True)
class BiasHLSBase(ModuleHLSBase):

    # hardware parameters
    filters: int
    data_t: FixedPoint = FixedPoint(16, 8)
    bias_t: FixedPoint = FixedPoint(32, 16)

    # class variables
    name: ClassVar[str] = "bias"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def functional_model(self, data, biases):

        # check input dimensionality
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(data.shape[-iter_space_len:] == self.input_iter_space[0])

        # add the bias term to the data
        return data + biases

@dataclass
class BiasHLS(BiasHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.filters] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.filters] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.filters, self.data_t.width, self.bias_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"   : [1],
            "FF"    : [1],
            "DSP"   : [0],
            "BRAM"  : [0],
        }

@dataclass
class BiasHLS3D(ModuleHLS3DBase, BiasHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.filters] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.filters] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.filters, self.data_t.width, self.bias_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"   : [1],
            "FF"    : [1],
            "DSP"   : [0],
            "BRAM"  : [0],
        }

