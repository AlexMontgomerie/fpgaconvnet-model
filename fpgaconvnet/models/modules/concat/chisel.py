from typing import ClassVar, Optional
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import int2bits, ModuleChiselBase, Port
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.modules.resources import ResourceModel, eval_resource_model, get_cached_resource_model

# DEFAULT_FITTER = NNLSHeuristicResourceFitter()

@dataclass(kw_only=True)
class ConcatChisel(ModuleChiselBase):

    # hardware parameters
    ports: int
    channels: list[int]
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16, 8))
    input_buffer_depth: list[int] = field(default_factory=list) # type: ignore
    output_buffer_depth: int = 2

    # class variables
    name: ClassVar[str] = "concat"
    register: ClassVar[bool] = True

    def __post_init__(self):
        assert(len(self.channels) == self.ports)

        if len(self.input_buffer_depth) == 0:
            self.input_buffer_depth = [0]*self.ports

        # call previous post init methods
        super().__post_init__()

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
        return [ [self.channels[i]] for i in range(self.ports) ]

    @property
    def output_iter_space(self) -> list[list[int]]:
        return [ [sum(self.channels)] ]

    @property
    def rate_in(self) -> list[float]:
        return [ self.channels[i]/float(sum(self.channels)) for i in range(self.ports) ]

    @property
    def rate_out(self) -> list[float]:
        return [ 1.0 ]

    def pipeline_depth(self) -> int:
        return sum(self.channels)

    def resource_parameters(self) -> list[int]:
        return [ sum(self.channels), self.streams, self.data_t.width,
                sum(self.input_buffer_depth), self.output_buffer_depth ]

    def functional_model(self, *data: np.ndarray) -> np.ndarray:

        # check input dimensionality
        assert len(data) == self.ports, f"Not enough input ports ({len(data)} != {self.ports})"

        # concatenate along the channel dimension
        return np.concatenate(data, axis=-1)

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
def _(m: ConcatChisel, rsc_type: str, model: ResourceModel) -> int:

    # check the correct resource type
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"

    # get the resource model
    match rsc_type:
        case "DSP":
            return 0
        case _:
            return model(m)

