import math
from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model

@dataclass(kw_only=True)
class PoolChisel(ModuleChiselBase):

    # hardware parameters
    kernel_size: list[int]
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    pool_type: str = "max"
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "pool"
    register: ClassVar[bool] = True

    # def __post_init__(self):

    #     # format kernel size as a 2 element list
    #     if isinstance(self.kernel_size, int):
    #         self.kernel_size = [self.kernel_size]*2
    #     elif isinstance(self.kernel_size, list):
    #         assert len(self.kernel_size) == 2, "Must specify two kernel dimensions"
    #     else:
    #         raise TypeError

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [1] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [1] ]

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams, *self.kernel_size],
            data_type=self.data_t,
            buffer_depth=self.input_buffer_depth,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
            data_type=self.data_t,
            buffer_depth=self.output_buffer_depth,
            name="io_out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return int(math.log(np.prod(self.kernel_size), 2))

    def resource_parameters(self) -> list[int]:
        return [ int(np.prod(self.kernel_size)), self.streams, self.data_t.width,
                self.input_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return super().resource_parameters_heuristics({
            "Logic_LUT"  : [
                self.kernel_size[0]*self.kernel_size[1],
                self.data_t.width*self.kernel_size[0]*self.kernel_size[1], # tree buffer
                self.data_t.width*int2bits(self.kernel_size[0]*self.kernel_size[1]), # tree buffer
                self.kernel_size[0],self.kernel_size[1], # input ready
                1,
            ],
            "LUT_RAM"  : [
                # queue_lutram_resource_model(
                #     int2bits(self.kernel_size[0]*self.kernel_size[1])+1, self.data_t.width), # buffer
                1,
            ],
            "LUT_SR"  : [0],
            "FF"   : [
                self.data_t.width, # output buffer
                self.data_t.width*self.kernel_size[0]*self.kernel_size[1], # op tree input
                int2bits(self.kernel_size[0]*self.kernel_size[1]), # shift register
                1,
            ],
            "DSP"  : [0],
            "BRAM36" : [0],
            "BRAM18" : [0],
        })


    def functional_model(self, *inputs: np.ndarray) -> np.ndarray:

        # get the input data
        data = inputs[0]

        # check input dimensions
        data_iter_space_len = len(self.input_iter_space[0]) + len(self.input_simd_lanes[0])
        data_iter_space = [*self.input_iter_space[0], *self.input_simd_lanes[0]]
        assert(len(data.shape) >= data_iter_space_len), \
                f"{len(data.shape)} is not greater than or equal to {data_iter_space_len}"
        assert(list(data.shape[-data_iter_space_len:]) == data_iter_space), \
                f"{list(data.shape[-data_iter_space_len:])} is not equal to {data_iter_space}"

        # perform the pooling operation
        last_dim =  tuple(-1 - np.arange(len(self.kernel_size)))
        match self.pool_type:
            case 'max':
                return np.max(data, axis=last_dim)
            case 'avg':
                return np.mean(data, axis=last_dim)
            case _:
                raise ValueError(f"Invalid pool type: {self.pool_type}")


@eval_resource_model.register
def _(m: PoolChisel, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    match rsc_type:
        case "DSP":
            return 0
        case "BRAM":
            return 0
        case _:
            return model(m)

