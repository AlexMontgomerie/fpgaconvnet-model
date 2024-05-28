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

    def functional_model(self, *data: np.ndarray) -> np.ndarray:

        # check input dimensionality
        assert len(data) == self.ports , "ERROR: invalid number of ports"

        # check input data shape
        for i in range(self.ports):
            data_iter_space_len = len(self.input_iter_space[i]) + len(self.input_simd_lanes[i])
            data_iter_space = [*self.input_iter_space[i], *self.input_simd_lanes[i]]
            assert(len(data[i].shape) >= data_iter_space_len), \
                    f"{len(data[i].shape)} is not greater than or equal to {data_iter_space_len}"
            assert(list(data[i].shape[-data_iter_space_len:]) == data_iter_space), \
                    f"{list(data[i].shape[-data_iter_space_len:])} is not equal to {data_iter_space}"

        # perform elment wise operation
        match self.eltwise_type:
            case "add":
                return np.sum(data, axis=0)
            case "mul":
                return np.prod(data, axis=0)
            case _:
                raise ValueError(f"Element-wise type {self.eltwise_type} not supported")

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

