from dataclasses import dataclass
from typing import ClassVar
import numpy as np

from fpgaconvnet.models.modules import Port, ModuleBaseMeta, ModuleHLSBase, ModuleHLS3DBase, int2bits
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.data_types import FixedPoint

@dataclass
class ForkHLSBase(ModuleHLSBase):

    # hardware parameters
    coarse: int
    kernel_size: list[int]
    data_t: FixedPoint = FixedPoint(16, 8)

    # class variables
    name: ClassVar[str] = "fork"
    register: ClassVar[bool] = False

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[*self.kernel_size],
            data_type=self.data_t,
            buffer_depth=0,
            name="in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.coarse, *self.kernel_size],
            data_type=self.data_t,
            buffer_depth=0,
            name="out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self):
        return 1

    def functional_model(self, data):

        # replicate for coarse streams
        return np.repeat(np.expand_dims(data, axis=-2), self.coarse, axis=-2)

@dataclass
class ForkHLS(ForkHLSBase):

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
        return [ self.rows, self.cols, self.channels,
                int(np.prod(self.kernel_size)), self.coarse, self.data_t.width ]

@dataclass
class ForkHLS3D(ModuleHLS3DBase, ForkHLSBase):

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
        return [ self.rows, self.cols, self.depth, self.channels,
                        int(np.prod(self.kernel_size)), self.coarse, self.data_t.width ]

