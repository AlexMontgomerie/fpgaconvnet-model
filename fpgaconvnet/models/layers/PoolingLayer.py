import math
from typing import Any, List, Union
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.data_types import FixedPoint

# from fpgaconvnet.models.modules import SlidingWindow
# from fpgaconvnet.models.modules import Pool
# from fpgaconvnet.models.layers import Layer

@dataclass(kw_only=True)
# class PoolingLayer(Layer):
class PoolingLayer:
    coarse: int = 1
    pool_type = 'max'
    kernel_rows: int = 2
    kernel_cols: int = 2
    stride_rows: int = 2
    stride_cols: int = 2
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    fine: int = 1
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # backend flag
        assert self.backend in ["hls", "chisel"], f"{self.backend} is an invalid backend"

        # regression model
        assert(self.regression_model in ["linear_regression", "xgboost"],
                f"{self.regression_model} is an invalid regression model")

        # init modules
        self.modules["sliding_window"] = SlidingWindow(self.rows_in(),
                self.cols_in(), self.channels_in()//self.coarse,
                self.kernel_size, self.stride, self.pad_top,
                self.pad_right, self.pad_bottom, self.pad_left, backend=self.backend, regression_model=self.regression_model)
        self.modules["pool"] = Pool(self.rows_out(), self.cols_out(),
                self.channels_out()//self.coarse, self.kernel_size, backend=self.backend, regression_model=self.regression_model)

        if self.backend == "chisel":
            self.data_packing = True
        elif self.backend == "hls":
            self.data_packing = False

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

            case "fine":
                assert(value in self.get_fine_feasible())
                super().__setattr__(name, value)

            case _:
                super().__setattr__(name, value)

    def rows_out(self) -> int:
        return self.modules["sliding_window"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window"].cols_out()

    @property
    def kernel_size(self) -> List[int]:
        return [ self.kernel_rows, self.kernel_cols ]

    @property
    def stride(self) -> List[int]:
        return [ self.stride_rows, self.stride_cols ]

    @property
    def pad(self) -> List[int]:
        return [
            self.pad_top,
            self.pad_left,
            self.pad_bottom,
            self.pad_right,
        ]

    @kernel_size.setter
    def kernel_size(self, val: List[int]) -> None:
        self.kernel_rows = val[0]
        self.kernel_cols = val[1]

    @stride.setter
    def stride(self, val: List[int]) -> None:
        self.stride_rows = val[0]
        self.stride_cols = val[1]

    @pad.setter
    def pad(self, val: List[int]) -> None:
        self.pad_top    = val[0]
        self.pad_right  = val[3]
        self.pad_bottom = val[2]
        self.pad_left   = val[1]

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.kernel_size.extend(self.kernel_size)
        parameters.kernel_rows  = self.kernel_rows
        parameters.kernel_cols  = self.kernel_cols
        parameters.stride.extend(self.stride)
        parameters.stride_rows  = self.stride_rows
        parameters.stride_cols  = self.stride_cols
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left

    def update(self):
        # sliding window
        self.modules['sliding_window'].rows     = self.rows_in()
        self.modules['sliding_window'].cols     = self.cols_in()
        self.modules['sliding_window'].channels = self.channels_in()//self.coarse
        self.modules['sliding_window'].data_width = self.data_t.width
        if self.data_packing:
            self.modules['sliding_window'].streams = self.coarse
        # pool
        self.modules['pool'].rows     = self.rows_out()
        self.modules['pool'].cols     = self.cols_out()
        self.modules['pool'].channels = self.channels_in()//self.coarse
        self.modules['pool'].data_width = self.data_t.width
        if self.data_packing:
            self.modules['pool'].streams = self.coarse

    def get_fine_feasible(self):
        return [1]

    def resource(self):

        sw_rsc      = self.modules['sliding_window'].rsc()
        pool_rsc    = self.modules['pool'].rsc()

        # Total
        if self.data_packing:
            return {
                "LUT"  :  sw_rsc['LUT'] +
                        pool_rsc['LUT'],
                "FF"   :  sw_rsc['FF'] +
                        pool_rsc['FF'],
                "BRAM" :  sw_rsc['BRAM'] +
                        pool_rsc['BRAM'],
                "DSP" :   sw_rsc['DSP'] +
                        pool_rsc['DSP']
            }
        else:
            return {
                "LUT"  :  sw_rsc['LUT']*self.coarse +
                        pool_rsc['LUT']*self.coarse,
                "FF"   :  sw_rsc['FF']*self.coarse +
                        pool_rsc['FF']*self.coarse,
                "BRAM" :  sw_rsc['BRAM']*self.coarse +
                        pool_rsc['BRAM']*self.coarse,
                "DSP" :   sw_rsc['DSP']*self.coarse +
                        pool_rsc['DSP']*self.coarse
            }

    def visualise(self, name):

        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightskyblue")

        # names
        slwin_name = [""]*self.coarse
        pool_name = [""]*self.coarse

        for i in range(self.coarse):
            # define names
            slwin_name[i] = "_".join([name, "sw", str(i)])
            pool_name[i] = "_".join([name, "pool", str(i)])
            # add nodes
            cluster.add_node(self.modules["sliding_window"].visualise(slwin_name[i]))
            cluster.add_node(self.modules["pool"].visualise(pool_name[i]))
            # add edges
            cluster.add_edge(pydot.Edge(slwin_name[i], pool_name[i]))

        return cluster, np.array(slwin_name).flatten().tolist(), np.array(pool_name).flatten().tolist()

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # instantiate pooling layer
        pooling_layer = torch.nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=self.pad[0])

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return pooling_layer(torch.from_numpy(data)).detach().numpy()
