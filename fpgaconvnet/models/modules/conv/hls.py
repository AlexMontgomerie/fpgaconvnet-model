from dataclasses import dataclass
from typing import ClassVar, Union
import numpy as np
import random

from fpgaconvnet.models.modules import Port, ModuleBaseMeta, ModuleHLSBase, ModuleHLS3DBase, int2bits
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.data_types import FixedPoint

@dataclass
class ConvHLSBase(ModuleHLSBase):

    # hardware parameters
    fine: int
    filters: int
    groups: int
    kernel_size: Union[list[int], int]
    data_t: FixedPoint = FixedPoint(16, 8)
    weight_t: FixedPoint = FixedPoint(16, 8)
    acc_t: FixedPoint = FixedPoint(32, 16)

    # class variables
    name: ClassVar[str] = "conv"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.data_t,
            buffer_depth=0,
            name="in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.acc_t,
            buffer_depth=0,
            name="out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ self.fine*self.groups / float(np.prod(self.kernel_size)*self.filters) ]

    @property
    def rate_out(self) -> list[float]:
        return [ self.fine / float(np.prod(self.kernel_size)) ]

    def pipeline_depth(self):
        return self.fine

    def functional_model(self,data,weights): # FIXME

        out = np.zeros((
            self.rows,
            self.cols,
            self.channels,
            int(self.filters/self.groups)
        ),dtype=float)

        for index,_ in np.ndenumerate(out):
            for k1 in range(self.kernel_size[0]):
                for k2 in range(self.kernel_size[1]):
                    out[index] += data[
                      index[0],index[1],index[2],k1,k2]*weights[
                      index[2],index[3],k1,k2]

        return out

@dataclass
class ConvHLS(ConvHLSBase):

    register: ClassVar[bool] = True

    def __post_init__(self):

        # format kernel size as a 2 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size]*2
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels, *self.kernel_size] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels, self.filters] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.groups, self.channels,
                self.filters, np.prod(self.kernel_size), self.fine,
                self.data_t.width, self.weight_t.width, self.accum_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"  : np.array([
                self.int2bits(self.filters),
                self.int2bits(self.cols*self.rows),
                self.int2bits(self.channels)
            ]),
            "FF"   : np.array([
                self.int2bits(self.filters),
                self.int2bits(self.cols*self.rows),
                self.int2bits(self.channels)
            ]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1])
        }

@dataclass
class ConvHLS3D(ModuleHLS3DBase, ConvHLSBase):

    register: ClassVar[bool] = True

    def __post_init__(self):

        # format kernel size as a 3 element list
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size]*3
        elif isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == 3, "Must specify three kernel dimensions"
        else:
            raise TypeError

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels, *self.kernel_size] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels, self.filters] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.groups, self.channels,
                self.filters, np.prod(self.kernel_size), self.fine,
                self.data_t.width, self.weight_t.width, self.accum_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"  : np.array([
                self.int2bits(self.filters),
                self.int2bits(self.cols*self.rows*self.depth),
                self.int2bits(self.channels)
            ]),
            "FF"   : np.array([
                self.int2bits(self.filters),
                self.int2bits(self.cols*self.rows*self.depth),
                self.int2bits(self.channels)
            ]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1])
        }

