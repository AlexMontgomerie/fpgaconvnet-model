from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
import pydot

from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.layers import MultiPortLayer
# from fpgaconvnet.models.modules import EltWise


@dataclass(kw_only=True)
class EltWiseLayer(MultiPortLayer):
    ports_in: int = 1
    coarse: int = 1
    op_type: str = "add"
    broadcast: bool = False
    acc_t: FixedPoint = field(default_factory=lambda: FixedPoint(32,16), init=True)
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

        self.mem_bw_in = [100.0/self.ports_in] * self.ports_in

        # init modules
        self.modules = {
            "eltwise" : EltWise(self.rows_in(), self.cols_in(),
                self.channels_in()//self.coarse, self.ports_in, eltwise_type=self.op_type, broadcast=self.broadcast, backend=self.backend, regression_model=self.regression_model),
        }

        # update the layer
        self.update()

    def __setattr__(self, name: str, value: Any) -> None:

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        match name:
            case "coarse":
                assert(value in self.get_coarse_in_feasible())
                super().__setattr__("coarse", value)
                self.update()
            case "coarse_in":
                assert(value in self.get_coarse_in_feasible())
                super().__setattr__("coarse_in", [value]*self.ports_in)
                self.update()
            case "coarse_out":
                assert(value in self.get_coarse_out_feasible())
                super().__setattr__("coarse_out", [value])
                self.update()
            case "op_type":
                assert(value in ["add", "mul"])
                super().__setattr__("op_type", value)
                self.update()
            case _:
                super().__setattr__(name, value)

    def get_operations(self):
        return self.rows_in()*self.cols_in()*self.channels_in()

    def update(self):
        # eltwise
        self.modules["eltwise"].rows     = self.rows_in()
        self.modules["eltwise"].cols     = self.cols_in()
        self.modules["eltwise"].channels = int(self.channels_in()/self.coarse)
        self.modules["eltwise"].ports_in   = self.ports_in
        self.modules["eltwise"].data_width = self.data_t.width

    def layer_info(self,parameters,batch_size=1):
        MultiPortLayer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        self.acc_t.to_protobuf(parameters.acc_t)
        # remove the repeated rows, cols and channels
        del parameters.rows_in_array[:]
        del parameters.cols_in_array[:]
        del parameters.channels_in_array[:]
        del parameters.rows_out_array[:]
        del parameters.cols_out_array[:]
        del parameters.channels_out_array[:]

    def resource(self):
        fifo_rsc = super().resource()
        eltwise_rsc = self.modules['eltwise'].rsc()

        # Total
        return {
            "LUT"  :  eltwise_rsc['LUT']*self.coarse,
            "FF"   :  eltwise_rsc['FF']*self.coarse,
            "BRAM" :  eltwise_rsc['BRAM']*self.coarse + fifo_rsc['BRAM'],
            "DSP" :   eltwise_rsc['DSP']*self.coarse
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightgrey")

        # names
        eltwise_name = [""]*self.coarse

        for i in range(self.coarse):
            # get the relu name
            eltwise_name[i] = "_".join([name, "eltwise", str(i)])
            # add nodes
            cluster.add_node(self.modules["eltwise"].visualise(eltwise_name[i]))

        return cluster, np.array(eltwise_name).tolist(), np.array(eltwise_name).tolist()

    def functional_model(self, data, batch_size=1):

        assert len(data) == self.ports_in
        for i in range(len(data)):
            assert data[i].shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
            assert data[i].shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
            assert data[i].shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # return output featuremap
        if self.op_type == "add":
            out = np.zeros((
                self.rows_out(),
                self.cols_out(),
                self.channels_out()),dtype=float)

            for index, _ in np.ndenumerate(out):
                for i in range(self.ports_in):
                    out[index] += float(data[i][index])
        elif self.op_type == "mul":
            out = np.ones((
                self.rows_out(),
                self.cols_out(),
                self.channels_out()),dtype=float)

            for index, _ in np.ndenumerate(out):
                for i in range(self.ports_in):
                    out[index] *= float(data[i][index])
        else:
            raise ValueError(f"Element-wise type {self.eltwise_type} not supported")

        # return output featuremap
        out = np.moveaxis(out, -1, 0)
        out = np.repeat(out[np.newaxis,...], batch_size, axis=0)
        return out

