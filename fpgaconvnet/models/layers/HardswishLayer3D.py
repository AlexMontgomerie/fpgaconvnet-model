from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pydot

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import Hardswish3D
from fpgaconvnet.models.layers import Layer3D

@dataclass(kw_only=True)
class HardswishLayer3D(Layer3D):
    coarse: int = 1
    input_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8), init=True)
    output_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8), init=True)
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # init modules
        self.modules["hardswish3d"] = Hardswish3D(self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in()//self.coarse, backend=self.backend, regression_model=self.regression_model)

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
        return self.rows_in()*self.cols_in()*self.depth_in()*self.channels_in()

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        self.input_t.to_protobuf(parameters.input_t)
        self.output_t.to_protobuf(parameters.output_t)

    def update(self):
        self.modules['hardswish3d'].rows     = self.rows_in()
        self.modules['hardswish3d'].cols     = self.cols_in()
        self.modules['hardswish3d'].depth    = self.depth_in()
        self.modules['hardswish3d'].channels = int(self.channels_in()/self.coarse)

    def resource(self):

        # get hardswish3d resources
        hardswish3d_rsc = self.modules['hardswish3d'].rsc()

        # Total
        return {
            "LUT"  :  hardswish3d_rsc['LUT']*self.coarse,
            "FF"   :  hardswish3d_rsc['FF']*self.coarse,
            "BRAM" :  hardswish3d_rsc['BRAM']*self.coarse,
            "DSP" :   hardswish3d_rsc['DSP']*self.coarse,
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightgrey")

        # names
        hardswish3d_name = [""]*self.coarse

        for i in range(self.coarse):
            # get the hardswish3d name
            hardswish3d_name[i] = "_".join([name, "hardswish3d", str(i)])
            # add nodes
            cluster.add_node(self.modules["hardswish3d"].visualise(hardswish3d_name[i]))

        return cluster, np.array(hardswish3d_name).tolist(), np.array(hardswish3d_name).tolist()

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth_in()   , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels_in(), "ERROR: invalid channel dimension"

        # instantiate hardswish layer
        hardswish_layer = torch.nn.Hardswish()

        # return output featuremap
        data = np.moveaxis(data, [-1, -2], [0, 1])
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return hardswish_layer(torch.from_numpy(data)).detach().numpy()

