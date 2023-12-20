import numpy as np
import math
import onnx
import pydot

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import Activation3D
from fpgaconvnet.models.layers import Layer3D

class ActivationLayer3D(Layer3D):
    def __init__(
            self,
            rows: int,
            cols: int,
            depth: int,
            channels: int,
            activation_type: str,
            coarse: int = 1,
            data_t: FixedPoint = FixedPoint(16,8),
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression",
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0]
        ):

        # initialise parent class
        super().__init__(rows, cols, depth, channels,
                coarse, coarse, data_t=data_t,
                input_compression_ratio=input_compression_ratio,
                output_compression_ratio=output_compression_ratio)
        # set the activation layer name based on the activation type
        self.layer_name = f"{activation_type}3d"

        # save parameters
        self._coarse = coarse

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        # init modules
        self.modules[self.layer_name] = Activation3D(self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in()//self.coarse, activation_type, backend=self.backend, regression_model=self.regression_model)

        self.update()

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
        self.coarse_out = val
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

