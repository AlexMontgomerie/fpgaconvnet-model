from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model


@dataclass(kw_only=True)
class SparseVectorMultiplyChisel(ModuleChiselBase):

    # hardware parameters
    coarse: int
    filters: int
    multipliers: int
    sparsity: list[int]
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    weight_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32, 16))
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

    def functional_model(self, *inputs: np.ndarray) -> np.ndarray:

        # get the input data
        data = inputs[0]
        weights = inputs[1]

        # check input dimensions
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(list(data.shape[-iter_space_len:]) == self.input_iter_space[0])

        iter_space_len = len(self.input_iter_space[1])
        assert(len(weights.shape) >= iter_space_len)
        assert(list(weights.shape[-iter_space_len:]) == self.input_iter_space[1])

        # unpack the inputs
        data, weights = inputs

        tmp = np.repeat(np.expand_dims(data, axis=-3), self.filters, axis=-3)
        return np.multiply(tmp, weights)


@eval_resource_model.register
def _(m: SparseVectorMultiplyChisel, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    return model(m)


