from typing import Any
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.data_types import FixedPoint
# from fpgaconvnet.models.modules import GlobalPool3D
from fpgaconvnet.models.layers import Layer3D

@dataclass(kw_only=True)
class GlobalPoolingLayer3D(Layer3D):
    coarse: int = 1
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32,16), init=True)
    op_type: str = "avg" # TODO: support different op types
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # backend flag
        assert (self.backend in ["hls", "chisel"], f"{self.backend} is an invalid backend")

        # regression model
        assert(self.regression_model in ["linear_regression", "xgboost"],
                f"{self.regression_model} is an invalid regression model")

        self.pool_type = self.op_type

        # init modules
        self.modules["global_pool3d"] = GlobalPool3D(
                self.rows_in(), self.cols_in(), self.depth_in(),
                self.channels_in()//self.coarse,
                backend=self.backend, regression_model=self.regression_model)

        self.update()

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

    def get_operations(self):
        return self.channels_in()*self.rows_in()*self.cols_in()*self.depth_in()

    def rows_out(self) -> int:
        return 1

    def cols_out(self) -> int:
        return 1

    def depth_out(self) -> int:
        return 1

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
