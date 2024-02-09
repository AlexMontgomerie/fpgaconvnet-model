import numpy as np
import math
import pydot
from typing import Union, List

from fpgaconvnet.data_types import FixedPoint

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
            data_t: FixedPoint = FixedPoint(16,8),
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression",
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0]
        ):

        # initialise parent class
        super().__init__([rows]*ports_in, [cols]*ports_in, channels,
                [coarse]*ports_in, [coarse]*ports_in, ports_in=ports_in,
                data_t=data_t,
                input_compression_ratio=input_compression_ratio,
                output_compression_ratio=output_compression_ratio)

        self.mem_bw_in = [100.0] * self.ports_in
        # parameters
        self._coarse = coarse

        # backend flag
        assert backend in ["chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        # init modules
        self.modules = {
            # "concat" : Concat(self.rows_in(), self.cols_in(), self.channels, self.ports_in,
            #     backend=self.backend, regression_model=self.regression_model),
            "concat" : Concat(self.rows_in(), self.cols_in(), self.channels, self.ports_in,
                backend=self.backend, regression_model=self.regression_model),
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

    def get_coarse_in_feasible(self, port_index=0):
        assert(port_index < self.ports_in)
        factors = set( get_factors(self.channels_in(0)) )
        for i in range(self.ports_in):
            factors &= set( get_factors(self.channels_in(i)) )
        return list(factors)

    def get_coarse_out_feasible(self, port_index=0):
        assert(port_index < self.ports_out)
        return self.get_coarse_in_feasible()

    def update(self):
        # concat
        self.modules["concat"].rows     = self.rows_in()
        self.modules["concat"].cols     = self.cols_in()
        self.modules["concat"].channels = [self.channels_in(i)//self.coarse for i in range(self.ports_in)]
        self.modules["concat"].ports_in = self.ports_in

    def layer_info(self,parameters,batch_size=1):
        MultiPortLayer.layer_info(self, parameters, batch_size)
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.coarse       = self._coarse
       # remove the repeated rows, cols and channels
        del parameters.rows_in_array[:]
        del parameters.cols_in_array[:]
        del parameters.rows_out_array[:]
        del parameters.cols_out_array[:]
        del parameters.channels_out_array[:]

