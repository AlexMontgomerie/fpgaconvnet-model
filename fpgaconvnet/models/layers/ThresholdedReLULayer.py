import torch
import numpy as np
import math
import onnx
import pydot

from fpgaconvnet.data_types import FixedPoint

# from fpgaconvnet.models.modules import ThresholdedReLU
# from fpgaconvnet.models.layers import Layer

# class ThresholdedReLULayer(Layer):
class ThresholdedReLULayer:
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            threshold: float = 0.0,
            coarse: int = 1,
            data_t: FixedPoint = FixedPoint(16,8),
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression"
        ):

        # initialise parent class
        super().__init__(rows, cols, channels,
                coarse, coarse, data_t=data_t)

        # save parameters
        self._threshold = threshold
        self._coarse = coarse

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        # init modules
        self.modules["relu"] = ThresholdedReLU(self.rows_in(), self.cols_in(),
                self.channels_in()//self.coarse, backend=self.backend, regression_model=self.regression_model)

        self.update()

    @property
    def threshold(self) -> float:
        return self._threshold

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
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.threshold = self.threshold

    def update(self):
        self.modules['relu'].rows     = self.rows_in()
        self.modules['relu'].cols     = self.cols_in()
        self.modules['relu'].channels = int(self.channels_in()/self.coarse)
        self.modules['relu'].threshold = self.threshold

    def resource(self):

        # get relu resources
        relu_rsc = self.modules['relu'].rsc()

        # Total
        return {
            "LUT"  :  relu_rsc['LUT']*self.coarse,
            "FF"   :  relu_rsc['FF']*self.coarse,
            "BRAM" :  relu_rsc['BRAM']*self.coarse,
            "DSP" :   relu_rsc['DSP']*self.coarse,
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightgrey")

        # names
        relu_name = [""]*self.coarse

        for i in range(self.coarse):
            # get the relu name
            relu_name[i] = "_".join([name, "relu", str(i)])
            # add nodes
            cluster.add_node(self.modules["relu"].visualise(relu_name[i]))

        return cluster, np.array(relu_name).tolist(), np.array(relu_name).tolist()

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR: invalid channel dimension"

        # instantiate relu layer
        relu_layer = torch.nn.Threshold(self.threshold, 0.0)

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return relu_layer(torch.from_numpy(data)).detach().numpy()

