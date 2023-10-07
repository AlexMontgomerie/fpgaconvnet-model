import math
from dataclasses import dataclass, field
from typing import Any, List, Union

import numpy as np
import onnx
import pydot

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import Hardswish
from fpgaconvnet.models.layers import Layer

@dataclass(kw_only=True)
class HardswishLayer(Layer):
    coarse: int = 1
    input_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8), init=True)
    output_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8), init=True)
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # init modules
        self.modules["hardswish"] = Hardswish(self.rows_in(), self.cols_in(),
                self.channels_in()//self.coarse, backend=self.backend, regression_model=self.regression_model)

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
        return self.rows_in()*self.cols_in()*self.channels_in()

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        self.input_t.to_protobuf(parameters.input_t)
        self.output_t.to_protobuf(parameters.output_t)

    def update(self):
        self.modules['hardswish'].rows     = self.rows_in()
        self.modules['hardswish'].cols     = self.cols_in()
        self.modules['hardswish'].channels = int(self.channels_in()/self.coarse)

    def resource(self):

        # get hardswish resources
        hardswish_rsc = self.modules['hardswish'].rsc()

        # Total
        return {
            "LUT"  :  hardswish_rsc['LUT']*self.coarse,
            "FF"   :  hardswish_rsc['FF']*self.coarse,
            "BRAM" :  hardswish_rsc['BRAM']*self.coarse,
            "DSP" :   hardswish_rsc['DSP']*self.coarse,
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightgrey")

        # names
        hardswish_name = [""]*self.coarse

        for i in range(self.coarse):
            # get the hardswish name
            hardswish_name[i] = "_".join([name, "hardswish", str(i)])
            # add nodes
            cluster.add_node(self.modules["hardswish"].visualise(hardswish_name[i]))

        return cluster, np.array(hardswish_name).tolist(), np.array(hardswish_name).tolist()

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR: invalid channel dimension"

        # instantiate hardswish layer
        hardswish_layer = torch.nn.Hardswish()

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return hardswish_layer(torch.from_numpy(data)).detach().numpy()

