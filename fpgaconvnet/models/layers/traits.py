from dataclasses import dataclass
from typing import ClassVar, Any

import numpy as np
import pydot
from dacite import from_dict
from google.protobuf.json_format import MessageToDict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.layers.utils import balance_module_rates, get_factors
from fpgaconvnet.architecture import BACKEND, DIMENSIONALITY
from fpgaconvnet.models.layers import LayerBase

@dataclass(kw_only=True)
class LayerMatchingCoarse(LayerBase):
    coarse: int = 1

    def __setattr__(self, name: str, value: Any) -> None:

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        try:
            match name:
                case "coarse" | "coarse_in" | "coarse_out":
                    assert(value in self.get_coarse_in_feasible())
                    assert(value in self.get_coarse_out_feasible())
                    super().__setattr__("coarse_in", value)
                    super().__setattr__("coarse_out", value)
                    super().__setattr__("coarse", value)
                    self.update()

                case _:
                    super().__setattr__(name, value)

        except AttributeError:
            print(f"WARNING: unable to set attribute {name}, trying super method")
            super().__setattr__(name, value)

    def streams(self) -> int:
        return self.coarse

@dataclass(kw_only=True)
class Layer2D(LayerBase):
    rows: int
    cols: int
    channels: int

    dimensionality: ClassVar[DIMENSIONALITY] = DIMENSIONALITY.TWO

    def rows_in(self) -> int:
        return self.rows

    def cols_in(self) -> int:
       return self.cols

    def channels_in(self) -> int:
       return self.channels

    def rows_out(self) -> int:
       return self.rows

    def cols_out(self) -> int:
       return self.cols

    def channels_out(self) -> int:
       return self.channels

    def build_rates_graph(self):

        # create the rates graph
        rates_graph = np.zeros(shape=(len(self.modules.keys()),
                                      len(self.modules.keys())+1) , dtype=float )

        # iterate over modules
        for i, module in enumerate(self.modules.keys()):
            # update rates_graph
            rates_graph[i,i] = self.modules[module].rate_in[0]
            rates_graph[i,i+1] = self.modules[module].rate_out[0]

        # return rates_graph
        return rates_graph

    def input_shape(self) -> list[int]:
        return [ self.rows_in(), self.cols_in(), self.channels_in() ]

    def output_shape(self) -> list[int]:
        return [ self.rows_out(), self.cols_out(), self.channels_out() ]

    def input_shape_dict(self) -> dict[str,int]:
        return {
            "rows": self.rows_in(),
            "cols": self.cols_in(),
            "channels": self.channels_in(),
        }

    def output_shape_dict(self) -> dict[str,int]:
        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "channels": self.channels_out(),
        }

    def get_coarse_in_feasible(self):
        return get_factors(self.channels_in())

    def get_coarse_out_feasible(self):
        return get_factors(self.channels_out())

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(self, parameters, batch_size)
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()


@dataclass(kw_only=True)
class Layer3D(Layer2D):
    depth: int

    dimensionality: ClassVar[DIMENSIONALITY] = DIMENSIONALITY.THREE

    def depth_in(self) -> int:
        return self.depth

    def depth_out(self) -> int:
       return self.depth

    def input_shape(self) -> list[int]:
        return [ self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in() ]

    def output_shape(self) -> list[int]:
        return [ self.rows_out(), self.cols_out(), self.depth_out(), self.channels_out() ]

    def input_shape_dict(self) -> dict[str,int]:
        return {
            "rows": self.rows_in(),
            "cols": self.cols_in(),
            "depth": self.depth_in(),
            "channels": self.channels_in(),
        }

    def output_shape_dict(self) -> dict[str,int]:
        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "depth": self.depth_out(),
            "channels": self.channels_out(),
        }

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(self, parameters, batch_size)
        parameters.depth_in     = self.depth_in()
        parameters.depth_out    = self.depth_out()

