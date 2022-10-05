import numpy as np
import math
import pydot
import torch
from typing import Union, List

from fpgaconvnet.models.modules import Concat
from fpgaconvnet.models.layers import MultiPortLayer

from fpgaconvnet.models.layers.utils import get_factors

class ConcatLayer(MultiPortLayer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: List[int],
            ports_in: int = 1,
            coarse: int = 1,
            data_width: int = 16
        ):

        # initialise parent class
        super().__init__([rows], [cols], channels, [coarse], [coarse],
                ports_in=ports_in, data_width=data_width)

        # parameters
        self._coarse = coarse

        # init modules
        self.modules = {
            "concat" : Concat(self.rows_in(), self.cols_in(), self.channels, self.ports_in),
        }

        # update the layer
        self.update()

    def channels_out(self, port_index=0):
        assert port_index == 0, "ConcatLayer only has a single output port"
        return sum(self.channels)

    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> int:
        return [self._coarse]*self.ports_in

    @property
    def coarse_out(self) -> int:
        return [self._coarse]

    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        self.update()

    def rates_in(self, port_index=0):
        assert port_index < self.ports_in
        return self.modules["concat"].rate_in(port_index)

    def rates_out(self, port_index=0):
        assert port_index == 0, "ConcatLayer only has a single output port"
        return self.modules["concat"].rate_out()

    def get_coarse_in_feasible(self, port_index=0, wr_factor=1):
        assert(port_index < self.ports_in)
        factors = set( get_factors(self.channels_in(0)/wr_factor) )
        for i in range(self.ports_in):
            factors &= set( get_factors(self.channels_in(i)/wr_factor) )
        return list(factors)

    def get_coarse_out_feasible(self, port_index=0, wr_factor=1):
        assert(port_index < self.ports_out)
        return self.get_coarse_in_feasible(wr_factor=wr_factor)

    def update(self):
        # concat
        self.modules["concat"].rows     = self.rows_in()
        self.modules["concat"].cols     = self.cols_in()
        self.modules["concat"].channels = self.channels
        self.modules["concat"].ports_in = self.ports_in
        self.modules["concat"].data_width = self.data_width



