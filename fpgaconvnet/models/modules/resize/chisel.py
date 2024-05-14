from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model


@dataclass(kw_only=True)
class ResizeChisel(ModuleChiselBase):

    # hardware parameters
    rows: int
    cols: int
    channels: int
    scales: list[int]
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(32, 16))
    ram_style: str = "distributed"
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "resize"
    register: ClassVar[bool] = True

    def __post_init__(self):
        assert len(self.scales) == 3, "ResizeChisel only supports 3D scaling"
        assert self.scales[2] == 1, "ResizeChisel currently does not support channel scaling"

    @property
    def rows_out(self) -> int:
        return self.rows*self.scales[0]

    @property
    def cols_out(self) -> int:
        return self.cols*self.scales[1]

    @property
    def channels_out(self) -> int:
        return self.channels*self.scales[2]

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows_out, self.cols_out, self.channels_out] ]

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
    def rate_in(self) -> list[float]:
        return [ 1.0 / float(np.prod(self.scales)) ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def resource_parameters(self) -> list[int]:
        ram_style_int = 0 if self.ram_style == "distributed" else 1 # TODO: use an enumeration instead
        return [ self.rows, self.cols, self.channels, self.streams, self.data_t.width,
                ram_style_int, self.input_buffer_depth, self.output_buffer_depth,
                *self.scales ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return super().resource_parameters_heuristics({
            "Logic_LUT" : [1],
            "LUT_RAM"   : [1],
            "LUT_SR"    : [0],
            "FF"        : [1],
            "DSP"       : [0],
            "BRAM36"    : [0],
            "BRAM18"    : [0],
        })


    def functional_model(self, *inputs: np.ndarray) -> np.ndarray:

        # get the input data
        data = inputs[0]

        # check input dimensions
        iter_space_len = len(self.input_iter_space[0])
        assert(len(data.shape) >= iter_space_len)
        assert(list(data.shape[-iter_space_len:]) == self.input_iter_space[0])

        # perform upsampling
        return data.repeat(self.scales[0], axis=-3).repeat(self.scales[1], axis=-2).repeat(self.scales[2], axis=-1)


@eval_resource_model.register
def _(m: ResizeChisel, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    return model(m)

