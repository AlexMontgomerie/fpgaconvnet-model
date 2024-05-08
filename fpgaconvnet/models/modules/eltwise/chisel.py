from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model


@dataclass(kw_only=True)
class EltwiseChisel(ModuleChiselBase):

    # hardware parameters
    ports: int
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32, 16))
    eltwise_type: str = "add"
    broadcast: bool = False
    input_buffer_depth: list[int] = field(default_factory=list) # type: ignore
    output_buffer_depth: int = 0

    # class variables
    name: ClassVar[str] = "eltwise"
    register: ClassVar[bool] = True

    def __post_init__(self):
        if len(self.input_buffer_depth) == 0:
            self.input_buffer_depth = [0]*self.ports

    @property
    def input_ports(self) -> list[Port]:
        return [ Port(
            simd_lanes=[self.streams],
            data_type=self.data_t,
            buffer_depth=self.input_buffer_depth[i],
            name=f"io_in_{i}"
        ) for i in range(self.ports) ]

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
        return [ [1] for i in range(self.ports) ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [1] ]

    @property
    def rate_in(self) -> list[float]:
        return [ 1.0 for i in range(self.ports) ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return 1

    def resource_parameters(self) -> list[int]:
        return [ self.streams, self.data_t.width,
                sum(self.input_buffer_depth), self.output_buffer_depth ]

    # def rsc(self, coef=None, model=None):
    #     """
    #     Returns
    #     -------
    #     dict
    #         estimated resource usage of the module. Uses the
    #         resource coefficients for the estimate.
    #     """
    #     # get the channel buffer BRAM estimate
    #     channel_buffer_bram = bram_array_resource_model(int(self.channels), self.data_width, "fifo")

    #     return {
    #         "LUT"   : 49,
    #         "FF"    : 23,
    #         "BRAM"  : channel_buffer_bram if self.broadcast else 0,
    #         "DSP"   : 0 if self.eltwise_type == "add" else 1
    #     }

    def functional_model(self, *data: np.ndarray) -> np.ndarray:

        # check input dimensionality
        assert len(data) == self.ports , "ERROR: invalid number of ports"

        # perform elment wise operation
        match self.eltwise_type:
            case "add":
                return np.sum(data, axis=0)
            case "mul":
                return np.prod(data, axis=0)
            case _:
                raise ValueError(f"Element-wise type {self.eltwise_type} not supported")


@eval_resource_model.register
def _(m: EltwiseChisel, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    match rsc_type:
        case "DSP":
            return 0
        case _:
            return model(m)

