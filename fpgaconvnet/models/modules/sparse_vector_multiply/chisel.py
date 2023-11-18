import random
from typing import ClassVar, List
from dataclasses import dataclass

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY

@dataclass(kw_only=True)
class SparseVectorMultiplyChisel(ModuleChiselBase):

    # hardware parameters
    coarse: int
    filters: int
    multipliers: int
    sparsity: list[int]
    data_t: FixedPoint = FixedPoint(16, 8)
    weight_t: FixedPoint = FixedPoint(16, 8)
    acc_t: FixedPoint = FixedPoint(32, 16)
    use_dsp: bool = True
    input_buffer_depth: int = 2
    weight_buffer_depth: int = 2
    output_buffer_depth: int = 2

    # class variables
    name: ClassVar[str] = "sparse_vector_multiply"
    register: ClassVar[bool] = True

    # def __post_init__(self):
    #     assert len(self.sparsity) == np.prod(self.kernel_size) + 1, "ERROR: invalid sparsity"

    @property
    def input_ports(self) -> list[Port]:
        return [
            Port(
                simd_lanes=[self.streams, self.coarse],
                data_type=self.data_t,
                buffer_depth=self.input_buffer_depth,
                name="io_in"
            ),
            Port(
                simd_lanes=[self.streams, self.coarse],
                data_type=self.weight_t,
                buffer_depth=self.weight_buffer_depth,
                name="io_weights"
            ),

        ]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams, self.coarse],
            data_type=self.acc_t,
            buffer_depth=self.output_buffer_depth,
            name="io_out"
        )]

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [1], [self.filters] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.filters] ]

    # @property
    # def rate_in(self) -> list[float]:
    #     return [ (1.0/float(self.filters))*self.rate_kernel_sparsity(), self.rate_kernel_sparsity() ]

    # @property
    # def rate_out(self) -> list[float]:
    #     return [ self.rate_kernel_sparsity() ]

    # def pipeline_depth(self) -> int:
    #     return self.filters*(self.channels-1)

    def resource_parameters(self) -> list[int]:
        return [ self.filters, self.streams, self.multipliers, self.coarse,
                self.input_buffer_depth, self.weight_buffer_depth, self.output_buffer_depth,
                self.data_t.width, self.weight_t.width, self.acc_t.width ]

    def functional_model(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:

        tmp = np.repeat(np.expand_dims(data, axis=-3), self.filters, axis=-3)
        return np.multiply(tmp, weights)

