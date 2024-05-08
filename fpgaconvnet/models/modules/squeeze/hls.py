from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleHLSBase, ModuleHLS3DBase, Port
from fpgaconvnet.models.modules.squeeze import lcm
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model


@dataclass(kw_only=True)
class SqueezeHLSBase(ModuleHLSBase):

    # hardware parameters
    coarse_in: int
    coarse_out: int
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))

    # class variables
    name: ClassVar[str] = "squeeze"
    register: ClassVar[bool] = False

    @property
    def buffer_size(self) -> int:
        return lcm(self.coarse_in, self.coarse_out)

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.coarse_in],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_in"
        )]

    @property
    def output_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.coarse_out],
            data_type=self.data_t,
            buffer_depth=2,
            name="io_out"
        )]

    @property
    def rate_in(self) -> list[float]:
        return [ self.coarse_in / max(self.coarse_in, self.coarse_out) ]

    @property
    def rate_out(self) -> list[float]:
        return [ self.coarse_out / max(self.coarse_in, self.coarse_out) ]

    def pipeline_depth(self) -> int:
        return lcm(self.coarse_in, self.coarse_out)


@dataclass
class SqueezeHLS(SqueezeHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.channels] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.channels, self.coarse_in,
                self.coarse_out, self.buffer_size, self.data_t.width ]

@dataclass
class SqueezeHLS3D(ModuleHLS3DBase, SqueezeHLSBase):

    register: ClassVar[bool] = True

    @property
    def input_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels] ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [self.rows, self.cols, self.depth, self.channels] ]

    def resource_parameters(self) -> list[int]:
        return [ self.rows, self.cols, self.depth, self.channels, self.coarse_in,
                self.coarse_out, self.buffer_size, self.data_t.width ]


@eval_resource_model.register
def _(m: SqueezeHLS, rsc_type: str, model: ResourceModel) -> int:

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

