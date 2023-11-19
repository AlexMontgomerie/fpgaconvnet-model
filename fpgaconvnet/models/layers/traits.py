import collections
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Any
import math

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
class Layer2D(LayerBase):
    rows: int
    cols: int
    channels: int

    dimensionality: ClassVar[DIMENSIONALITY] = DIMENSIONALITY.TWO

    def __post_init__(self):
        super().__post_init__()

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
            rates_graph[i,i] = self.modules[module].rate_in()
            rates_graph[i,i+1] = self.modules[module].rate_out()

        # return rates_graph
        return rates_graph

    def rate_in(self) -> float:
        return abs(balance_module_rates(self.build_rates_graph())[0,0])

    def rate_out(self) -> float:
        return abs(balance_module_rates(
            self.build_rates_graph())[len(self.modules.keys())-1,len(self.modules.keys())])

    def shape_in(self) -> list[int]: # TODO: add documentation
        return [ self.rows_in(), self.cols_in(), self.channels_in() ]

    def shape_out(self) -> list[int]: # TODO: add documentation
        return [ self.rows_out(), self.cols_out(), self.channels_out() ]

    def width_in(self):
        return self.data_t.width

    def width_out(self):
       return self.data_t.width

    def get_coarse_in_feasible(self):
        return get_factors(int(self.channels_in()))

    def get_coarse_out_feasible(self):
        return get_factors(int(self.channels_out()))

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(self, parameters, batch_size)
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()

    # def visualise(self,name):
    #     cluster = pydot.Cluster(name,label=name)

    #     for i in range(self.coarse_in):
    #         cluster.add_node(pydot.Node( "_".join([name,"edge",str(i)]), label=self.__class__.__name__ ))

    #     return cluster, "_".join([name,"edge"]), "_".join([name,"edge"])

@dataclass(kw_only=True)
class Layer3D(Layer2D):
    depth: int

    dimensionality: ClassVar[DIMENSIONALITY] = DIMENSIONALITY.THREE

    def depth_in(self) -> int:
        return self.depth

    def depth_out(self) -> int:
       return self.depth

    def shape_in(self) -> list[int]: # TODO: add documentation
        return [ self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in() ]

    def shape_out(self) -> list[int]: # TODO: add documentation
        return [ self.rows_out(), self.cols_out(), self.depth_out(), self.channels_out() ]

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(self, parameters, batch_size)
        parameters.depth_in     = self.depth_in()
        parameters.depth_out    = self.depth_out()


@dataclass(kw_only=True)
class LayerMatchingCoarse(LayerBase):
    coarse: int = 1

    def __setattr__(self, name: str, value: Any) -> None:

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

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

