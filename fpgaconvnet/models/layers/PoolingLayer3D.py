from typing import Any, List
from dataclasses import dataclass

import numpy as np
import pydot

# from fpgaconvnet.models.modules import SlidingWindow3D
# from fpgaconvnet.models.modules import Pool3D
# from fpgaconvnet.models.modules import Pad3D

# from fpgaconvnet.models.layers import Layer3D

@dataclass(kw_only=True)
# class PoolingLayer3D(Layer3D):
class PoolingLayer3D:
    coarse: int = 1
    pool_type = 'max'
    kernel_rows: int = 2
    kernel_cols: int = 2
    kernel_depth: int = 2
    stride_rows: int = 2
    stride_cols: int = 2
    stride_depth: int = 2
    pad_top: int = 0
    pad_right: int = 0
    pad_front: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_back: int = 0
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
        self.modules["pad3d"] = Pad3D(
                self.rows_in(), self.cols_in(), self.depth_in(),
                self.channels_in()//self.coarse,
                self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
                self.pad_front, self.pad_back, backend=self.backend,
                regression_model=self.regression_model)

        self.modules["sliding_window3d"] = SlidingWindow3D(
                self.rows_in() +self.pad_top + self.pad_bottom,
                self.cols_in() + self.pad_left + self.pad_right,
                self.depth_in() + self.pad_front + self.pad_back,
                self.channels_in()//self.coarse,
                self.kernel_rows, self.kernel_cols, self.kernel_depth,
                self.stride_rows, self.stride_cols, self.stride_depth,
                0, 0, 0, 0, 0, 0, backend=self.backend,
                regression_model=self.regression_model)

        self.modules["pool3d"] = Pool3D(
                self.rows_out(), self.cols_out(), self.depth_out(),
                self.channels_out()//self.coarse, self.kernel_rows,
                self.kernel_cols, self.kernel_depth,
                pool_type=self.pool_type, backend=self.backend,
                regression_model=self.regression_model)

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
        return self.modules["sliding_window3d"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window3d"].cols_out()

    def depth_out(self) -> int:
        return self.modules["sliding_window3d"].depth_out()

    @property
    def kernel_size(self) -> List[int]:
        return [ self._kernel_rows, self._kernel_cols, self._kernel_depth ]

    @property
    def stride(self) -> List[int]:
        return [ self._stride_rows, self._stride_cols, self._stride_depth ]

    @property
    def pad(self) -> List[int]:
        return [
            self._pad_top,
            self._pad_left,
            self._pad_front,
            self._pad_bottom,
            self._pad_right,
            self._pad_back,
        ]

    @kernel_size.setter
    def kernel_size(self, val: List[int]) -> None:
        assert(len(val) == 3, "kernel size must be a list of three integers")
        self.kernel_rows = val[0]
        self.kernel_cols = val[1]
        self.kernel_depth = val[2]

    @stride.setter
    def stride(self, val: List[int]) -> None:
        assert(len(val) == 3, "stride must be a list of three integers")
        self.stride_rows = val[0]
        self.stride_cols = val[1]
        self.stride_depth = val[2]

    @pad.setter
    def pad(self, val: List[int]) -> None:
        assert(len(val) == 6, "pad must be a list of six integers")
        self.pad_top    = val[0]
        self.pad_right  = val[4]
        self.pad_bottom = val[3]
        self.pad_left   = val[1]
        self.pad_front  = val[2]
        self.pad_back   = val[5]

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.kernel_size.extend(self.kernel_size)
        parameters.kernel_rows = self.kernel_rows
        parameters.kernel_cols = self.kernel_cols
        parameters.kernel_depth = self.kernel_depth
        parameters.stride.extend(self.stride)
        parameters.stride_rows = self.stride_rows
        parameters.stride_cols = self.stride_cols
        parameters.stride_depth = self.stride_depth
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_front    = self.pad_front
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left
        parameters.pad_back     = self.pad_back

    def update(self):

        # pad
        self.modules['pad3d'].rows     = self.rows
        self.modules['pad3d'].cols     = self.cols
        self.modules['pad3d'].depth    = self.depth
        self.modules['pad3d'].channels = self.channels//self.coarse
        self.modules['pad3d'].data_width = self.data_t.width
        self.modules['pad3d'].streams = self.coarse
        self.modules['pad3d'].pad_top = self.pad_top
        self.modules['pad3d'].pad_bottom = self.pad_bottom
        self.modules['pad3d'].pad_left = self.pad_left
        self.modules['pad3d'].pad_right = self.pad_right
        self.modules['pad3d'].pad_front = self.pad_front
        self.modules['pad3d'].pad_back = self.pad_back

        # sliding window
        self.modules['sliding_window3d'].rows     = self.rows + self.pad_top + self.pad_bottom
        self.modules['sliding_window3d'].cols     = self.cols + self.pad_left + self.pad_right
        self.modules['sliding_window3d'].depth    = self.depth + self.pad_front + self.pad_back
        self.modules['sliding_window3d'].channels = self.channels//self.coarse
        self.modules['sliding_window3d'].kernel_cols = self.kernel_cols
        self.modules['sliding_window3d'].kernel_rows = self.kernel_rows
        self.modules['sliding_window3d'].kernel_depth= self.kernel_depth
        self.modules['sliding_window3d'].stride_cols = self.stride_cols
        self.modules['sliding_window3d'].stride_rows = self.stride_rows
        self.modules['sliding_window3d'].stride_depth= self.stride_depth
        self.modules['sliding_window3d'].data_width = self.data_t.width
        if self.data_packing:
            self.modules['sliding_window3d'].streams = self.coarse
        self.modules['sliding_window3d'].pad_top = 0
        self.modules['sliding_window3d'].pad_bottom = 0
        self.modules['sliding_window3d'].pad_left = 0
        self.modules['sliding_window3d'].pad_right = 0
        self.modules['sliding_window3d'].pad_front = 0
        self.modules['sliding_window3d'].pad_back = 0

        # pool 3d
        self.modules['pool3d'].rows     = self.rows_out()
        self.modules['pool3d'].cols     = self.cols_out()
        self.modules['pool3d'].depth    = self.depth_out()
        self.modules['pool3d'].channels = int(self.channels_in()/self.coarse)
        self.modules['pool3d'].data_width = self.data_t.width
        if self.data_packing:
            self.modules['pool3d'].streams = self.coarse

    def get_fine_feasible(self):
        return [1]

    def resource(self):

        sw_rsc      = self.modules['sliding_window3d'].rsc()
        pool_rsc    = self.modules['pool3d'].rsc()

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
            slwin_name[i] = "_".join([name, "sw3d", str(i)])
            pool_name[i] = "_".join([name, "pool3d", str(i)])
            # add nodes
            cluster.add_node(self.modules["sliding_window3d"].visualise(slwin_name[i]))
            cluster.add_node(self.modules["pool3d"].visualise(pool_name[i]))
            # add edges
            cluster.add_edge(pydot.Edge(slwin_name[i], pool_name[i]))

        return cluster, np.array(slwin_name).flatten().tolist(), np.array(pool_name).flatten().tolist()

    def functional_model(self,data,batch_size=1):
        import torch

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.depth_in()   , "ERROR (data): invalid depth dimension"
        assert data.shape[3] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # instantiate pooling layer
        if self.pool_type == "max":
            pooling_layer = torch.nn.MaxPool3d((self.kernel_depth, self.kernel_rows, self.kernel_cols), stride=(self.stride_depth, self.stride_rows, self.stride_cols), padding=(self.pad_front, self.pad_top, self.pad_right))
        elif self.pool_type == "avg":
            pooling_layer = torch.nn.AvgPool3d((self.kernel_depth, self.kernel_rows, self.kernel_cols), stride=(self.stride_depth, self.stride_rows, self.stride_cols), padding=(self.pad_front, self.pad_top, self.pad_right))

        # return output featuremap
        data = np.moveaxis(data, [-1, -2], [0, 1])
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return pooling_layer(torch.from_numpy(data)).detach().numpy()
