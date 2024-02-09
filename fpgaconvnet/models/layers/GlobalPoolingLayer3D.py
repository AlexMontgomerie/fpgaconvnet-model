import math
from typing import Union, List

import numpy as np
import pydot

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.modules import GlobalPool3D
from fpgaconvnet.models.layers import Layer3D

class GlobalPoolingLayer3D(Layer3D):

    def __init__(
            self,
            rows: int,
            cols: int,
            depth: int,
            channels: int,
            coarse: int = 1,
            data_t: FixedPoint = FixedPoint(16,8),
            acc_t: FixedPoint = FixedPoint(32,16),
            op_type: str = "avg", # TODO: support different op types
            backend: str = "chisel",
            regression_model: str = "linear_regression",
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0]
        ):

        # save acc_t
        self.acc_t = acc_t

        # initialise parent class
        super().__init__(rows, cols, depth, channels,
        coarse, coarse, data_t=data_t,
        input_compression_ratio=input_compression_ratio,
        output_compression_ratio=output_compression_ratio)

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        # update parameters
        self._coarse = coarse

        self.pool_type = op_type

        # init modules
        self.modules["global_pool3d"] = GlobalPool3D(
                self.rows_in(), self.cols_in(), self.depth_in(),
                self.channels_in()//self.coarse,
                backend=self.backend, regression_model=self.regression_model)

        self.update()

    def get_operations(self):
        return self.channels_in()*self.rows_in()*self.cols_in()*self.depth_in()

    def rows_out(self) -> int:
        return 1

    def cols_out(self) -> int:
        return 1

    def depth_out(self) -> int:
        return 1

    def start_depth(self):
        return self.rows*self.cols*self.depth*self.channels//self.streams_in()

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
        # self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        # self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        # self.update()

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        self.acc_t.to_protobuf(parameters.acc_t)

    def update(self):
        # pool
        self.modules['global_pool3d'].rows     = self.rows_in()
        self.modules['global_pool3d'].cols     = self.cols_in()
        self.modules['global_pool3d'].depth    = self.depth_in()
        self.modules['global_pool3d'].channels = int(self.channels_in()/self.coarse)
        self.modules['global_pool3d'].data_width = self.data_t.width
        self.modules['global_pool3d'].acc_width = self.acc_t.width

    def resource(self):

        pool_rsc = self.modules['global_pool3d'].rsc()

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
            pool_name[i] = "_".join([name, "global_pool3d", str(i)])
            # add nodes
            cluster.add_node(self.modules["global_pool3d"].visualise(pool_name[i]))

        return cluster, np.array(pool_name).flatten().tolist(), np.array(pool_name).flatten().tolist()

    def functional_model(self, data, batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.depth_in()   , "ERROR (data): invalid depth dimension"
        assert data.shape[3] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # return output featuremap
        data = np.moveaxis(data, [-1, -2], [0, 1])
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return np.average(data, axis=(2,3,4))
