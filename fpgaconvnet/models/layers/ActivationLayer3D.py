import numpy as np
from typing import Any
from dataclasses import dataclass, field
import pydot

from fpgaconvnet.models.modules import Activation3D
from fpgaconvnet.models.layers import Layer3D

@dataclass(kw_only=True)
class ActivationLayer3D(Layer3D):
    activation_type: str
    coarse: int = 1
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

        # set the activation layer name based on the activation type
        self.layer_name = f"{self.activation_type}3d"

        # init modules
        self.modules[self.layer_name] = Activation3D(self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in()//self.coarse, self.activation_type, backend=self.backend, regression_model=self.regression_model)

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

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse

    def update(self):
        self.modules[self.layer_name].rows     = self.rows_in()
        self.modules[self.layer_name].cols     = self.cols_in()
        self.modules[self.layer_name].depth    = self.depth_in()
        self.modules[self.layer_name].channels = int(self.channels_in()/self.coarse)

    def resource(self):

        # get activation resources
        activation_rsc = self.modules[self.layer_name].rsc()

        # Total
        return {
            "LUT"  :  activation_rsc['LUT']*self.coarse,
            "FF"   :  activation_rsc['FF']*self.coarse,
            "BRAM" :  activation_rsc['BRAM']*self.coarse,
            "DSP" :   activation_rsc['DSP']*self.coarse,
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightgrey")

        # names
        activation_name = [""]*self.coarse

        for i in range(self.coarse):
            # get the activation name
            activation_name[i] = "_".join([name, self.layer_name, str(i)])
            # add nodes
            cluster.add_node(self.modules[self.layer_name].visualise(activation_name[i]))

        return cluster, np.array(activation_name).tolist(), np.array(activation_name).tolist()

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth_in()   , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels_in(), "ERROR: invalid channel dimension"

        # instantiate activation layer
        if self.activation_type == "relu":
            activation_layer = torch.nn.ReLU()
        elif self.activation_type == "sigmoid":
            activation_layer = torch.nn.Sigmoid()
        elif self.activation_type == "silu":
            activation_layer = torch.nn.SiLU()
        else:
            raise NotImplementedError(f"Activation type {self.activation_type} not implemented")

        # return output featuremap
        data = np.moveaxis(data, [-1, -2], [0, 1])
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return activation_layer(torch.from_numpy(data)).detach().numpy()

