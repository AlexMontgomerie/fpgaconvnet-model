import random
from typing import ClassVar, Union
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleHLSBase, ModuleHLS3DBase, Port

@dataclass(kw_only=True)
class PoolHLSBase(ModuleHLSBase):

    # hardware parameters
    kernel_size: Union[list[int], int]
    pool_t: FixedPoint = FixedPoint(16, 8)

    # class variables
    name: ClassVar[str] = "pool"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[*self.kernel_size],
            data_type=self.pool_t,
            buffer_depth=2,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[1],
            data_type=self.pool_t,
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

        # check input dimensions
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(data.shape[-iter_space_len] == self.input_iter_space[0])

        # TODO: flatten last two dimensions

        # perform the pooling operation
        match self.pool_type:
            case 'max':
                return np.max(data, axis=-1)
            case 'avg':
                return np.mean(data, axis=-1)
            case _:
                raise ValueError(f"Invalid pool type: {self.pool_type}")


@dataclass
class PoolHLS(PoolHLSBase):

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
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.channels, np.prod(self.kernel_size), self.pool_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"   : np.array([1]),
            "FF"    : np.array([1]),
            "DSP"   : np.array([0]),
            "BRAM"  : np.array([0]),
        }

@dataclass
class PoolHLS3D(ModuleHLS3DBase, PoolHLSBase):

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
        return [ [self.rows, self.cols, self.depth, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.channels, np.prod(self.kernel_size), self.pool_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "LUT"   : np.array([1]),
            "FF"    : np.array([1]),
            "DSP"   : np.array([0]),
            "BRAM"  : np.array([0]),
        }

