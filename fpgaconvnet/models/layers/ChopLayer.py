"""
The chop/fork/branch layer.
Takes one stream input and outputs several streams using the fork module.
"""

from typing import List
import pydot
import numpy as np
import os
import math

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import Fork
from fpgaconvnet.models.layers import MultiPortLayer

class ChopLayer(MultiPortLayer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            split: list[int],
            coarse: int = 1,
            ports_out: int = 1,
            data_t: FixedPoint = FixedPoint(16,8),
            backend: str = "chisel",
            regression_model: str = "linear_regression"
        ):

        # save split parameters
        self.split = split

        # initialise parent class
        super().__init__([rows], [cols], [channels], [coarse], [coarse],
                ports_out=ports_out, data_t=data_t)

        self.mem_bw_out = [100.0/self.ports_out] * self.ports_out

        # backend flag
        assert backend in ["chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        # parameters
        self._coarse = coarse

        # init modules
        #One fork module, fork coarse_out corresponds to number of layer output ports
        # self.modules["fork"] = Fork( self.rows_in(), self.cols_in(),
        #         self.channels_in(), 1, self.ports_out, backend=self.backend, regression_model=self.regression_model)

        # update the modules
        self.update()

    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> int:
        return [self._coarse]

    @property
    def coarse_out(self) -> int:
        return [self._coarse]*self.ports_out

    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = [val]
        self.coarse_out = [val]*self.ports_out
        # self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = [val]
        self._coarse_out = [val]*self.ports_out
        # self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = [val]
        self._coarse_out = [val]*self.ports_out
        # self.update()

    def streams_in(self, port_index=0):
        """
        Returns
        -------
        int
            number of parallel streams into the layer.
        """
        assert(port_index < self.ports_in)
        return self._coarse

    def rows_out(self, port_index=0) -> int:
        return self.rows[0]

    def cols_out(self, port_index=0) -> int:
        return self.cols[0]

    def channels_out(self, port_index=0) -> int:
        return self.channels[port_index]

    def rate_in(self, port_index=0):
        assert(port_index < self.ports_in)
        return 1.0

    def rate_out(self, port_index=0):
        assert(port_index < self.ports_out)
        return self.channels_out(port_index)/self.channels_in()

    def layer_info(self,parameters,batch_size=1):
        MultiPortLayer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        # remove the repeated rows, cols and channels
        del parameters.rows_in_array[:]
        del parameters.cols_in_array[:]
        del parameters.channels_in_array[:]
        del parameters.rows_out_array[:]
        del parameters.cols_out_array[:]

    def update(self):
        pass

    def resource(self):

        #Total
        return {
            "LUT"   : 0,
            "FF"    : 0,
            "BRAM"  : 0,
            "DSP"   : 0
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in self.coarse_in:
            cluster.add_node(pydot.Node( "_".join([name,"chop",str(i)]), label="chop" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"chop",str(i)]) for i in range(self.coarse) ]
        nodes_out = [ "_".join([name,"chop",str(i)]) for i in range(self.ports_out) ]

        return cluster, nodes_in, nodes_out


