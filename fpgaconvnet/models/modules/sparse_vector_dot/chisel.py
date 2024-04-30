from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model
from fpgaconvnet.platform import DEFAULT_CHISEL_PLATFORM


@dataclass(kw_only=True)
class SparseVectorDotChisel(ModuleChiselBase):

    # hardware parameters
    fine: int
    filters: int
    kernel_size: list[int]
    sparsity: list[int]
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    weight_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32, 16))
    use_dsp: bool = True
    input_buffer_depth: int = 0
    weight_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "sparse_vector_dot"
    register: ClassVar[bool] = True

    def __post_init__(self):
        assert len(self.kernel_size) == 2, "ERROR: invalid kernel size"
        assert len(self.sparsity) == np.prod(self.kernel_size) + 1, "ERROR: invalid sparsity"

    def rate_kernel_sparsity(self):
        cycles_per_bin = np.ceil(np.flip(np.arange(np.prod(self.kernel_size) + 1))/self.fine)
        cycles_per_bin[-1] = 1.0
        rate_per_stream = 1.0 / np.sum(cycles_per_bin*self.sparsity, axis = 1)
        return min(rate_per_stream)

    @property
    def input_ports(self) -> list[Port]:
        return [
            Port(
                simd_lanes=[self.streams, int(np.prod(self.kernel_size))],
                data_type=self.data_t,
                buffer_depth=self.input_buffer_depth,
                name="io_in"
            ),
            Port(
                simd_lanes=[self.streams, int(np.prod(self.kernel_size))],
                data_type=self.weight_t,
                buffer_depth=self.weight_buffer_depth,
                name="io_weights"
            ),

        ]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
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
        return [ (1.0/float(self.filters))*self.rate_kernel_sparsity(), self.rate_kernel_sparsity() ]

    @property
    def rate_out(self) -> list[float]:
        return [ self.rate_kernel_sparsity() ]

    # def pipeline_depth(self) -> int:
    #     return self.filters*(self.channels-1)

    def resource_parameters(self) -> list[int]:
        return [ self.filters, self.streams, self.fine, int(np.prod(self.kernel_size)),
                self.input_buffer_depth, self.weight_buffer_depth, self.output_buffer_depth,
                self.data_t.width, self.weight_t.width, self.acc_t.width ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "Logic_LUT" : [
                self.fine, self.data_t.width, self.weight_t.width,
                self.data_t.width*self.fine,
                self.weight_t.width*self.fine,
                self.acc_t.width*self.fine, # adder tree
                self.filters, # ready logic
                int2bits(self.filters), # filter counter
                1,
            ],
            "LUT_RAM"   : [
                # queue_lutram_resource_model(
                #     int2bits(self.fine)+3, self.acc_t.width), # buffer
                1,
            ],
            "LUT_SR"    : [
                int2bits(self.fine)+1, # tree buffer valid
            ],
            "FF"    : [
                self.acc_t.width, # output buffer TODO
                int2bits(self.filters), # filter counter
                int2bits(self.fine)+1, # tree buffer valid
                self.acc_t.width*self.fine, # adder tree reg
                # self.acc_t.width*(2**(int2bits(self.fine))), # tree buffer registers
                # self.acc_t.width*int2bits(self.fine), # tree buffer
                1,
            ],
            "DSP"       : [self.fine],
            "BRAM36"    : [0],
            "BRAM18"    : [0],
        }

    # def functional_model(self, data: np.ndarray) -> np.ndarray:

    #     # check input dimensions
    #     iter_space_len = len(self.input_iter_space[0])
    #     assert(len(data.shape) >= iter_space_len)
    #     assert(data.shape[-iter_space_len] == self.input_iteration_space(0))

    #     # accumulate across the channel dimension
    #     return np.sum(data, axis=-3)

try:
    DEFAULT_SLIDING_WINDOW_RSC_MODELS: dict[str, ResourceModel] = { rsc_type: get_cached_resource_model(SparseVectorDotChisel,
                                    rsc_type, "default") for rsc_type in DEFAULT_CHISEL_PLATFORM.resource_types }
except FileNotFoundError:
    print("CRITICAL WARNING: default resource models not found for SparseVectorDot, default resource modelling will fail")

@eval_resource_model.register
def _(m: SparseVectorDotChisel, rsc_type: str, _model: Optional[ResourceModel] = None) -> int:

    # get the resource model
    model: ResourceModel = _model if _model is not None else DEFAULT_SLIDING_WINDOW_RSC_MODELS[rsc_type]

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    return model(m)

