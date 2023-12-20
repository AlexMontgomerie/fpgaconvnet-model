import numpy as np
import math
import onnx
import pydot

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import Hardswish3D
from fpgaconvnet.models.layers import Layer3D

class HardswishLayer3D(Layer3D):
    def __init__(
            self,
            rows: int,
            cols: int,
            depth: int,
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
        super().__init__(rows, cols, depth, channels,
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
        self.modules["hardswish3d"] = Hardswish3D(self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in()//self.coarse, backend=self.backend, regression_model=self.regression_model)

        self.update()

    def get_operations(self):
        return self.rows_in()*self.cols_in()*self.depth_in()*self.channels_in()

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
        self.input_t.to_protobuf(parameters.input_t)
        self.output_t.to_protobuf(parameters.output_t)

    def update(self):
        self.modules['hardswish3d'].rows     = self.rows_in()
        self.modules['hardswish3d'].cols     = self.cols_in()
        self.modules['hardswish3d'].depth    = self.depth_in()
        self.modules['hardswish3d'].channels = int(self.channels_in()/self.coarse)

    def resource(self):

        # get hardswish resources
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
            # get the hardswish name
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

