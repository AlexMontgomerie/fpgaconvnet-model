from dataclasses import dataclass
from typing import ClassVar, Any
from abc import abstractmethod

import numpy as np
import pydot # type: ignore
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
                    print(f"WARNING: setting {name} to {value}")
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

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.coarse = self.coarse


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

    def input_shape(self, port_idx: int = 0) -> list[int]:
        assert port_idx == 0
        return [ self.rows_in(), self.cols_in(), self.channels_in() ]

    def output_shape(self, port_idx: int = 0) -> list[int]:
        assert port_idx == 0
        return [ self.rows_out(), self.cols_out(), self.channels_out() ]

    def input_shape_dict(self, port_idx: int = 0) -> dict[str,int]:
        assert port_idx == 0
        return {
            "rows": self.rows_in(),
            "cols": self.cols_in(),
            "channels": self.channels_in(),
        }

    def output_shape_dict(self, port_idx: int = 0) -> dict[str,int]:
        assert port_idx == 0
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
        super().layer_info(parameters, batch_size)
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

    def input_shape(self, port_idx: int = 0) -> list[int]:
        assert port_idx == 0
        return [ self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in() ]

    def output_shape(self, port_idx: int = 0) -> list[int]:
        assert port_idx == 0
        return [ self.rows_out(), self.cols_out(), self.depth_out(), self.channels_out() ]

    def input_shape_dict(self, port_idx: int = 0) -> dict[str,int]:
        assert port_idx == 0
        return {
            "rows": self.rows_in(),
            "cols": self.cols_in(),
            "depth": self.depth_in(),
            "channels": self.channels_in(),
        }

    def output_shape_dict(self, port_idx: int = 0) -> dict[str,int]:
        assert port_idx == 0
        return {
            "rows": self.rows_out(),
            "cols": self.cols_out(),
            "depth": self.depth_out(),
            "channels": self.channels_out(),
        }

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.depth_in     = self.depth_in()
        parameters.depth_out    = self.depth_out()


@dataclass(kw_only=True)
class MultiPortLayer2D(LayerBase):
    rows: list[int]
    cols: list[int]
    channels: list[int]

    ports_in: int = 1
    ports_out: int = 1

    dimensionality: ClassVar[DIMENSIONALITY] = DIMENSIONALITY.TWO

    @abstractmethod
    def rows_in(self, port_idx: int = 0) -> int:
        pass
        # assert port_idx < self.ports_in
        # return self.rows[port_idx]

    @abstractmethod
    def cols_in(self, port_idx: int = 0) -> int:
        pass
        # assert port_idx < self.ports_in
        # return self.cols[port_idx]

    @abstractmethod
    def channels_in(self, port_idx: int = 0) -> int:
        pass
        # assert port_idx < self.ports_in
        # return self.channels[port_idx]

    @abstractmethod
    def rows_out(self, port_idx: int = 0) -> int:
        pass
        # assert port_idx < self.ports_out
        # return self.rows[port_idx]

    @abstractmethod
    def cols_out(self, port_idx: int = 0) -> int:
        pass
        # assert port_idx < self.ports_out
        # return self.cols[port_idx]

    @abstractmethod
    def channels_out(self, port_idx: int = 0) -> int:
        pass
        # assert port_idx < self.ports_out
        # return self.channels[port_idx]

    def input_shape(self, port_idx: int = 0) -> list[int]:
        assert port_idx < self.ports_in
        return [ self.rows_in(port_idx), self.cols_in(port_idx), self.channels_in(port_idx) ]

    def output_shape(self, port_idx: int = 0) -> list[int]:
        assert port_idx < self.ports_out
        return [ self.rows_out(port_idx), self.cols_out(port_idx), self.channels_out(port_idx) ]

    def input_shape_dict(self, port_idx: int = 0) -> dict[str,int]:
        assert port_idx < self.ports_in
        return {
            "rows": self.rows_in(port_idx),
            "cols": self.cols_in(port_idx),
            "channels": self.channels_in(port_idx),
        }

    def output_shape_dict(self, port_idx: int = 0) -> dict[str,int]:
        assert port_idx < self.ports_out
        return {
            "rows": self.rows_out(port_idx),
            "cols": self.cols_out(port_idx),
            "channels": self.channels_out(port_idx),
        }

    def get_coarse_in_feasible(self) -> list[int]:
        coarse_in_feasible = set(get_factors(self.channels_in(0)))
        for i in range(1, self.ports_in):
            coarse_in_feasible = coarse_in_feasible.intersection(set(get_factors(self.channels_in(i))))
        return list(coarse_in_feasible)

    def get_coarse_out_feasible(self) -> list[int]:
        coarse_out_feasible = set(get_factors(self.channels_out(0)))
        for i in range(1, self.ports_in):
            coarse_out_feasible = coarse_out_feasible.intersection(set(get_factors(self.channels_out(i))))
        return list(coarse_out_feasible)

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(parameters, batch_size)
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.ports_in = self.ports_in
        parameters.ports_out = self.ports_out


