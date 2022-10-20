import numpy as np
import math
import pydot
import torch
from typing import Union, List

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import EltWise
from fpgaconvnet.models.layers import MultiPortLayer

from fpgaconvnet.models.layers.utils import get_factors

class EltWiseLayer(MultiPortLayer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            ports_in: int = 1,
            coarse: int = 1,
            op_type: str = "sum",
            acc_t: FixedPoint = FixedPoint(32,16),
        ):

        # initialise parent class
        super().__init__([rows]*ports_in, [cols]*ports_in,
                [channels]*ports_in, [coarse]*ports_in,
                [coarse]*ports_in, ports_in=ports_in)

        self.acc_t = acc_t

        # parameters
        self._coarse = coarse
        self._op_type = op_type

        # init modules
        self.modules = {
            "eltwise" : EltWise(self.rows_in(), self.cols_in(),
                self.channels_in(), self.ports_in),
        }

        # update the layer
        self.update()

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

    def update(self):
        # eltwise
        self.modules["eltwise"].rows     = self.rows_in()
        self.modules["eltwise"].cols     = self.cols_in()
        self.modules["eltwise"].channels = self.channels_in()
        self.modules["eltwise"].ports_in = self.ports_in
        self.modules["eltwise"].data_width = self.data_t.width

    def layer_info(self,parameters,batch_size=1):
        MultiPortLayer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        self.acc_t.to_protobuf(parameters.acc_t)
        # remove the repeated rows, cols and channels
        del parameters.rows_in_array[:]
        del parameters.cols_in_array[:]
        del parameters.channels_in_array[:]
        del parameters.rows_out_array[:]
        del parameters.cols_out_array[:]
        del parameters.channels_out_array[:]

    def functional_model(self, data, batch_size=1):

        assert len(data) == self.ports_in
        for i in range(len(data)):
            assert data[i].shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
            assert data[i].shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
            assert data[i].shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # return output featuremap
        if self._op_type == "sum":
            return sum(data)

