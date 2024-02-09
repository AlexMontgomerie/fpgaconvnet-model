import numpy as np
import math
import onnx
import pydot
from typing import Union, List

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import ReSize
from fpgaconvnet.models.layers import Layer

class ReSizeLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            scales: List[int],
            coarse: int = 1,
            data_t: FixedPoint = FixedPoint(16,8),
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression",
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0]
        ):

        # initialise parent class
        super().__init__(rows, cols, channels,
                coarse, coarse, data_t=data_t,
                input_compression_ratio=input_compression_ratio,
                output_compression_ratio=output_compression_ratio)

        # save parameters
        self._coarse = coarse
        self.scales = scales

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        # init modules
        self.modules["resize"] = ReSize(self.rows_in(), self.cols_in(),
                self.channels_in()//self.coarse, scales=self.scales,
                backend=self.backend, regression_model=self.regression_model)

        self.update()

    def get_operations(self):
        return self.rows_in()*self.cols_in()*self.channels_in()

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

    def rows_out(self) -> int:
        return self.modules['resize'].rows_out()

    def cols_out(self) -> int:
        return self.modules['resize'].cols_out()

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.scale.extend(self.scales)

    def update(self):
        self.modules['resize'].rows     = self.rows_in()
        self.modules['resize'].cols     = self.cols_in()
        self.modules['resize'].channels = self.channels_in()//self.coarse
        self.modules['resize'].scales = self.scales

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

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR: invalid channel dimension"

        # instantiate resize layer
        resize_layer = torch.nn.ReLU()

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return resize_layer(torch.from_numpy(data)).detach().numpy()

