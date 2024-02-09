import numpy as np
import math
import onnx
import pydot

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import Hardswish
from fpgaconvnet.models.layers import Layer

class HardswishLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int = 1,
            input_t: FixedPoint = FixedPoint(16,8),
            output_t: FixedPoint = FixedPoint(16,8),
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression",
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0]
        ):

        # initialise parent class
        super().__init__(rows, cols, channels,
                coarse, coarse, data_t=input_t,
                input_compression_ratio=input_compression_ratio,
                output_compression_ratio=output_compression_ratio)

        # save parameters
        self._coarse = coarse

        # save data types
        self.input_t = input_t
        self.output_t = output_t

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        # init modules
        self.modules["hardswish"] = Hardswish(self.rows_in(), self.cols_in(),
                self.channels_in()//self.coarse, backend=self.backend, regression_model=self.regression_model)

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

