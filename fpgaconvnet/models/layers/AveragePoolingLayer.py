import math
from typing import Union, List

import torch
import numpy as np
import pydot

from fpgaconvnet.models.modules import AveragePool
from fpgaconvnet.models.layers import Layer

class AveragePoolingLayer(Layer):

    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int = 1,
            data_width: int = 16
        ):

        # initialise parent class
        super().__init__(rows, cols, channels, coarse,
                coarse, data_width=data_width)

        # update flags
        # self.flags['transformable'] = True

        # update parameters
        self._coarse = coarse

        # init modules
        self.modules["average_pool"] = AveragePool(
                self.rows_in(), self.cols_in(),
                int(self.channels_in()/self.coarse))

        self.update()

    def rows_out(self) -> int:
        return 1

    def cols_out(self) -> int:
        return 1

    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> int:
        return self._coarse

    @property
    def coarse_out(self) -> int:
        return self._coarse

    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse

    def update(self):
        # pool
        self.modules['average_pool'].rows     = self.rows_in()
        self.modules['average_pool'].cols     = self.cols_in()
        self.modules['average_pool'].channels = int(self.channels_in()/self.coarse)

    def resource(self):

        pool_rsc = self.modules['average_pool'].rsc()

        # Total
        return {
            "LUT"  :  pool_rsc['LUT']*self.coarse,
            "FF"   :  pool_rsc['FF']*self.coarse,
            "BRAM" :  pool_rsc['BRAM']*self.coarse,
            "DSP" :   pool_rsc['DSP']*self.coarse
        }

    def visualise(self, name):

        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightskyblue")

        # names
        pool_name = [""]*self.coarse

        for i in range(self.coarse):
            # define names
            pool_name[i] = "_".join([name, "pool", str(i)])
            # add nodes
            cluster.add_node(self.modules["pool"].visualise(pool_name[i]))

        return cluster, np.array(pool_name).flatten().tolist(), np.array(pool_name).flatten().tolist()

    def functional_model(self, data, batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return np.average(data, axis=(2,3))
