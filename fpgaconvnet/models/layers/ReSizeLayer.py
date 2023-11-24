from dataclasses import dataclass
from typing import Any, List

import numpy as np
import onnx
import pydot

# from fpgaconvnet.data_types import FixedPoint
# from fpgaconvnet.models.layers import Layer
# from fpgaconvnet.models.modules import ReSize


@dataclass(kw_only=True)
# class ReSizeLayer(Layer):
class ReSizeLayer:
    scales: List[int]
    mode: str = "nearest"
    coarse: int = 1

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        assert self.mode in ["nearest", "linear", "bilinear", "trilinear"], f"ERROR: invalid resize mode '{self.mode}'"

        # init modules
        self.modules["resize"] = ReSize(self.rows_in(), self.cols_in(),
                self.channels_in()//self.coarse, scales=self.scales, mode=self.mode, backend=self.backend, regression_model=self.regression_model)

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

    def rows_out(self) -> int:
        return self.modules['resize'].rows_out()

    def cols_out(self) -> int:
        return self.modules['resize'].cols_out()

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.scale.extend(self.scales)
        parameters.mode = self.mode

    def update(self):
        self.modules['resize'].rows     = self.rows_in()
        self.modules['resize'].cols     = self.cols_in()
        self.modules['resize'].channels = self.channels_in()//self.coarse
        self.modules['resize'].scales = self.scales
        self.modules['resize'].mode = self.mode

    def resource(self):

        # # get resize resources
        # resize_rsc = self.modules['resize'].rsc()

        # Total
        return {
            "LUT"  :  0,
            "FF"   :  0,
            "BRAM" :  0,
            "DSP" :   0,
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightgrey")

        # names
        resize_name = [""]*self.coarse

        for i in range(self.coarse):
            # get the resize name
            resize_name[i] = "_".join([name, "resize", str(i)])
            # add nodes
            cluster.add_node(self.modules["resize"].visualise(resize_name[i]))

        return cluster, np.array(resize_name).tolist(), np.array(resize_name).tolist()

