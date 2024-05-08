from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model


@dataclass(kw_only=True)
class StrideChisel(ModuleChiselBase):

    # hardware parameters
    rows: int
    cols: int
    channels: int
    kernel_size: list[int]
    stride: list[int]
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(32, 16))
    input_buffer_depth: int = 0
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "stride"
    register: ClassVar[bool] = True

    def __post_init__(self):
        assert len(self.kernel_size) == 2, "StrideChisel only supports 3D kernel size"
        assert len(self.stride) == 2, "StrideChisel only supports 3D stride"

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows // self.stride[0], self.cols // self.stride[1], self.channels] ]

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
            simd_lanes=[self.streams, *self.kernel_size],
            data_type=self.data_t,
            buffer_depth=self.output_buffer_depth,
            name="io_out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 / float(np.prod(self.stride)) ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.channels, self.streams, int(np.prod(self.kernel_size)),
                *self.stride, self.data_t.width, self.input_buffer_depth, self.output_buffer_depth ]

    def resource_parameters_heuristics(self) -> dict[str, list[int]]:
        return {
            "Logic_LUT" : [1],
            "LUT_RAM"   : [1],
            "LUT_SR"    : [0],
            "FF"        : [1],
            "DSP"       : [0],
            "BRAM36"    : [0],
            "BRAM18"    : [0],
        }


    def functional_model(self, data):
        # check input dimensionality
        # assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.kernel_size[0]  , "ERROR: invalid column dimension"
        assert data.shape[4] == self.kernel_size[1]  , "ERROR: invalid column dimension"

        out = np.ndarray((
            self.rows//self.stride[0],
            self.cols//self.stride[1],
            self.channels,
            self.kernel_size[0],
            self.kernel_size[1]),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0]*self.stride[0],
              index[1]*self.stride[1],
              index[2],index[3],index[4]]

        return out

@eval_resource_model.register
def _(m: StrideChisel, rsc_type: str, model: ResourceModel) -> int:

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

