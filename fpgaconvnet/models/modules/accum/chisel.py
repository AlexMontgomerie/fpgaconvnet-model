from typing import ClassVar, Optional
from dataclasses import dataclass, field


import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model

@dataclass(kw_only=True)
class AccumChisel(ModuleChiselBase):

    # hardware parameters
    channels: int
    filters: int
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(32, 16))
    ram_style: str = "distributed"
    input_buffer_depth: int = 0
    acc_buffer_depth: int = 3
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "accum"
    register: ClassVar[bool] = True

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
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
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.channels, self.filters] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.filters] ]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0/float(self.channels) ]

    def pipeline_depth(self) -> int:
        return self.filters*(self.channels-1)

    def resource_parameters(self) -> list[int]:
        ram_style_int = 0 if self.ram_style == "distributed" else 1 # TODO: use an enumeration instead
        return [ self.channels, self.filters, self.streams, self.data_t.width, ram_style_int,
                self.input_buffer_depth, self.acc_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return super().resource_parameters_heuristics({
                "Logic_LUT" : [
                    self.filters, self.channels, # parameter logic
                    self.streams*self.data_t.width, # input word logic
                    self.streams, # input streams logic
                    int2bits(self.channels), # channel cntr
                    int2bits(self.filters), # filter cntr
                    1, # extra
                ],
                "LUT_RAM" : [
                    self.streams*self.data_t.width*self.filters, # filter memory memory (size)
                    self.streams*self.data_t.width, # filter memory memory (word width)
                    self.filters, # filter memory memory (depth)
                ],
                "LUT_SR" : [0],
                "FF" : [
                    self.data_t.width,  # input val cache
                    self.streams*self.data_t.width,  # input val cache
                    int2bits(self.channels), # channel_cntr
                    int2bits(self.filters), # filter cntr
                    self.channels, # channel parameter reg
                    self.filters, # filter parameter reg
                    1, # other registers
                ],
                "DSP" : [0],
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

        # accumulate across the channel dimension
        return np.sum(data, axis=-3)


@eval_resource_model.register
def _(m: AccumChisel, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # return zero if channels are 1
    if m.channels == 1:
        return 0

    # get the resource model
    match rsc_type:
        case "DSP":
            return 0
        case _:
            return model(m)

