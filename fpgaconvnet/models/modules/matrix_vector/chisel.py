from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model

@dataclass(kw_only=True)
class MatrixVectorChisel(ModuleChiselBase):

    # hardware parameters
    fine: int
    coarse: int
    filters: int
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    weight_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32, 16))
    use_dsp: bool = True
    input_buffer_depth: int = 2
    weight_buffer_depth: int = 2
    output_buffer_depth: int = 2

    # class variables
    name: ClassVar[str] = "matrix_vector"
    register: ClassVar[bool] = True

    @property
    def input_ports(self) -> list[Port]:
        return [
            Port(
                simd_lanes=[self.streams, self.fine],
                data_type=self.data_t,
                buffer_depth=self.input_buffer_depth,
                name="io_in"
            ),
            Port(
                simd_lanes=[self.streams, self.fine, self.coarse],
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

    @property
    def rate_in(self) -> list[float]:
        return [ (1.0/float(self.filters)), 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    # def pipeline_depth(self) -> int:
    #     return self.filters*(self.channels-1)

    def resource_parameters(self) -> list[int]:
        return [ self.filters, self.streams, self.fine, self.coarse,
                self.input_buffer_depth, self.weight_buffer_depth, self.output_buffer_depth,
                self.data_t.width, self.weight_t.width, self.acc_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return super().resource_parameters_heuristics({
            "Logic_LUT" : [
                self.fine, self.data_t.width, self.weight_t.width,
                self.streams*self.data_t.width*self.fine,
                self.streams*self.weight_t.width*self.fine,
                self.streams*self.acc_t.width*self.fine, # adder tree
                self.filters, # ready logic
                int2bits(self.filters), # filter counter
                1,
            ],
            "LUT_RAM"   : [
                # queue_lutram_resource_model(
                #     int2bits(self.fine)+1, self.streams*self.acc_t.width), # buffer
                1,
            ],
            "LUT_SR"    : [
                int2bits(self.fine)+1, # tree buffer valid
            ],
            "FF"    : [
                self.acc_t.width, # output buffer TODO
                self.filters, self.fine, # parameters
                int2bits(self.filters), # filter counter
                int2bits(self.fine)+1, # tree buffer valid
                self.streams*self.acc_t.width*self.fine, # adder tree reg
                self.streams*self.acc_t.width, # output buffer
                1,
            ],
            "DSP"       : [self.streams*self.fine],
            "BRAM36"    : [0],
            "BRAM18"    : [0],
        })

    def functional_model(self, *inputs: np.ndarray) -> np.ndarray:

        # get the input data
        data = inputs[0]
        weights = inputs[1]

        # check input dimensions
        data_iter_space_len = len(self.input_iter_space[0]) + len(self.input_simd_lanes[0])
        data_iter_space = [*self.input_iter_space[0], *self.input_simd_lanes[0]]
        assert(len(data.shape) >= data_iter_space_len), \
                f"{len(data.shape)} is not greater than or equal to {data_iter_space_len}"
        assert(list(data.shape[-data_iter_space_len:]) == data_iter_space), \
                f"{list(data.shape[-data_iter_space_len:])} is not equal to {data_iter_space}"

        # check weight dimensions
        weight_iter_space_len = len(self.input_iter_space[1]) + len(self.input_simd_lanes[1])
        weight_iter_space = [*self.input_iter_space[1], *self.input_simd_lanes[1]]
        assert(len(weights.shape) >= weight_iter_space_len), \
                f"{len(weights.shape)} is not greater than or equal to {weight_iter_space_len}"
        assert(list(weights.shape[-weight_iter_space_len:]) == weight_iter_space), \
                f"{list(weights.shape[-weight_iter_space_len:])} is not equal to {weight_iter_space}"

        # replicate for the coarse dimension
        data = np.repeat(np.expand_dims(data, axis=-1), self.coarse, axis=-1)

        # replicate for filter dimension
        partial = np.repeat(data, self.filters, axis=-4)

        # multiply weights and data
        partial = np.multiply(partial, weights)

        # sum across the fine dimension
        return np.sum(partial, axis=-2)


@eval_resource_model.register
def _(m: MatrixVectorChisel, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    return model(m)

if __name__ == "__main__":

    m = MatrixVectorChisel(fine=4, coarse=8, filters=16)
    data = np.random.rand(10, *m.input_iter_space[0], *m.input_simd_lanes[0])
    weights = np.random.rand(10, *m.input_iter_space[1], *m.input_simd_lanes[1])
    m.functional_model(data, weights)

