import numpy as np
import math
import pydot
from typing import Union, List

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import EltWise3D
from fpgaconvnet.models.layers import MultiPortLayer3D

from fpgaconvnet.models.layers.utils import get_factors

class EltWiseLayer3D(MultiPortLayer3D):
    def __init__(
            self,
            rows: int,
            cols: int,
            depth: int,
            channels: int,
            ports_in: int = 1,
            coarse: int = 1,
            op_type: str = "add",
            broadcast: bool = False,
            data_t: FixedPoint = FixedPoint(16,8),
            acc_t: FixedPoint = FixedPoint(32,16),
            backend: str = "chisel", # default to no bias for old configs
            regression_model: str = "linear_regression",
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0]
        ):

        # initialise parent class
        super().__init__([rows]*ports_in, [cols]*ports_in,
                [depth]*ports_in,
                [channels]*ports_in, [coarse]*ports_in,
                [coarse], ports_in=ports_in, data_t=data_t,
                input_compression_ratio=input_compression_ratio,
                output_compression_ratio=output_compression_ratio)

        self.mem_bw_in = [100.0/self.ports_in] * self.ports_in

        self.acc_t = acc_t

        # parameters
        self._coarse = coarse
        self._op_type = op_type
        self._broadcast = broadcast

        # backend flag
        assert backend in ["chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        # init modules
        self.modules = {
            "eltwise3d" : EltWise3D(self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in()//self.coarse, self.ports_in, eltwise_type=op_type, broadcast=broadcast, backend=self.backend, regression_model=self.regression_model),
        }

        # update the layer
        self.update()

    def get_operations(self):
        return self.channels_in()*self.rows_in()*self.cols_in()*self.depth_in()

    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> int:
        return [self._coarse]*self.ports_in

    @property
    def coarse_out(self) -> int:
        return [self._coarse]

    @property
    def op_type(self) -> int:
        return self._op_type

    @property
    def broadcast(self) -> bool:
        return self._broadcast

    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        # self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        # self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        # self.update()

    @op_type.setter
    def op_type(self, val: str) -> None:
        self._op_type = val
        # self.update()

    @broadcast.setter
    def broadcast(self, val: bool) -> None:
        self._broadcast = val
        # self.update()

    def update(self):
        # eltwise
        self.modules["eltwise3d"].rows     = self.rows_in()
        self.modules["eltwise3d"].cols     = self.cols_in()
        self.modules["eltwise3d"].depth    = self.depth_in()
        self.modules["eltwise3d"].channels = int(self.channels_in()/self.coarse)
        self.modules["eltwise3d"].ports_in   = self.ports_in
        self.modules["eltwise3d"].data_width = self.data_t.width

    def layer_info(self,parameters,batch_size=1):
        MultiPortLayer3D.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.depth_in     = self.depth_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.depth_out    = self.depth_out()
        parameters.channels_out = self.channels_out()
        self.acc_t.to_protobuf(parameters.acc_t)
        # remove the repeated rows, cols and channels
        del parameters.rows_in_array[:]
        del parameters.cols_in_array[:]
        del parameters.depth_in_array[:]
        del parameters.channels_in_array[:]
        del parameters.rows_out_array[:]
        del parameters.cols_out_array[:]
        del parameters.depth_out_array[:]
        del parameters.channels_out_array[:]

    def resource(self):
        fifo_rsc = super().resource()
        eltwise_rsc = self.modules['eltwise3d'].rsc()

        # Total
        return {
            "LUT"  :  eltwise_rsc['LUT']*self.coarse,
            "FF"   :  eltwise_rsc['FF']*self.coarse,
            "BRAM" :  eltwise_rsc['BRAM']*self.coarse + fifo_rsc['BRAM'],
            "DSP" :   eltwise_rsc['DSP']*self.coarse,
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightgrey")

        # names
        eltwise_name = [""]*self.coarse

        for i in range(self.coarse):
            # get the relu name
            eltwise_name[i] = "_".join([name, "eltwise3d", str(i)])
            # add nodes
            cluster.add_node(self.modules["eltwise3d"].visualise(eltwise_name[i]))

        return cluster, np.array(eltwise_name).tolist(), np.array(eltwise_name).tolist()

    def functional_model(self, data, batch_size=1):

        assert len(data) == self.ports_in
        for i in range(len(data)):
            assert data[i].shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
            assert data[i].shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
            assert data[i].shape[2] == self.depth_in()   , "ERROR (data): invalid depth dimension"
            assert data[i].shape[3] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # return output featuremap
        if self.op_type == "add":
            out = np.zeros((
                self.rows_out(),
                self.cols_out(),
                self.depth_out(),
                self.channels_out()),dtype=float)

            for index, _ in np.ndenumerate(out):
                for i in range(self.ports_in):
                    out[index] += float(data[i][index])
        elif self.op_type == "mul":
            out = np.ones((
                self.rows_out(),
                self.cols_out(),
                self.depth_out(),
                self.channels_out()),dtype=float)
            for index, _ in np.ndenumerate(out):
                for i in range(self.ports_in):
                    out[index] *= float(data[i][index])
        else:
            raise ValueError(f"Element-wise type {self.eltwise_type} not supported")

        # return output featuremap
        out = np.moveaxis(out, [-1, -2], [0, 1])
        out = np.repeat(out[np.newaxis,...], batch_size, axis=0)
        return out
