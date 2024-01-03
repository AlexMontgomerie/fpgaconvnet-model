import math
from typing import Union, List

import numpy as np
import pydot

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import SlidingWindow3D
from fpgaconvnet.models.modules import Pool3D
from fpgaconvnet.models.modules import Pad3D

from fpgaconvnet.models.layers import Layer3D

class PoolingLayer3D(Layer3D):

    def __init__(
            self,
            rows: int,
            cols: int,
            depth: int,
            channels: int,
            coarse: int = 1,
            pool_type   ='max',
            kernel_rows: int = 1,
            kernel_cols: int = 1,
            kernel_depth: int = 1,
            stride_rows: int = 1,
            stride_cols: int = 1,
            stride_depth: int = 1,
            pad_top: int = 0,
            pad_right: int = 0,
            pad_front: int = 0,
            pad_bottom: int = 0,
            pad_left: int = 0,
            pad_back: int = 0,
            fine: int = 1,
            data_t: FixedPoint = FixedPoint(16,8),
            backend: str = "chisel",
            regression_model: str = "linear_regression",
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0]
        ):

        # initialise parent class
        super().__init__(rows, cols, depth, channels,
                coarse, coarse, data_t=data_t,
                input_compression_ratio=input_compression_ratio,
                output_compression_ratio=output_compression_ratio)

        # update parameters
        self._kernel_rows = kernel_rows
        self._kernel_cols = kernel_cols
        self._kernel_depth = kernel_depth
        self._stride_rows = stride_rows
        self._stride_cols = stride_cols
        self._stride_depth = stride_depth
        self._pad_top = pad_top
        self._pad_right = pad_right
        self._pad_front = pad_front
        self._pad_bottom = pad_bottom
        self._pad_left = pad_left
        self._pad_back = pad_back
        self._pool_type = pool_type
        self._coarse = coarse
        self._fine = fine

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        # init modules
        if self.backend == "chisel":
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
        elif self.backend == "hls":
            self.modules["sliding_window3d"] = SlidingWindow3D(
                    self.rows_in(),
                    self.cols_in(),
                    self.depth_in(),
                    self.channels_in()//self.coarse,
                    self.kernel_rows, self.kernel_cols, self.kernel_depth,
                    self.stride_rows, self.stride_cols, self.stride_depth,
                    self.pad_top, self.pad_right, self.pad_front, self.pad_bottom,
                    self.pad_left, self.pad_back, backend=self.backend,
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

    def get_operations(self):
        return self.channels_in()*self.rows_out()*self.cols_out()*self.depth_out()\
            *self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]

    def rows_out(self) -> int:
        return self.modules["sliding_window3d"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window3d"].cols_out()

    def depth_out(self) -> int:
        return self.modules["sliding_window3d"].depth_out()

    def start_depth(self):
        return (self.kernel_rows-1-self.pad_top)*self.cols*self.depth*self.channels//self.streams_in() + \
            (self.kernel_cols-1-self.pad_left)*self.depth*self.channels//self.streams_in() + \
            (self.kernel_depth-1-self.pad_front)*self.channels//self.streams_in() + \
            self.channels//self.streams_in()

    @property
    def kernel_size(self) -> List[int]:
        return [ self._kernel_rows, self._kernel_cols, self._kernel_depth ]

    @property
    def kernel_rows(self) -> int:
        return self._kernel_rows

    @property
    def kernel_cols(self) -> int:
        return self._kernel_cols

    @property
    def kernel_depth(self) -> int:
        return self._kernel_depth

    @property
    def stride(self) -> List[int]:
        return [ self._stride_rows, self._stride_cols, self._stride_depth ]

    @property
    def stride_rows(self) -> int:
        return self._stride_rows

    @property
    def stride_cols(self) -> int:
        return self._stride_cols

    @property
    def stride_depth(self) -> int:
        return self._stride_depth

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

    @property
    def pad_top(self) -> int:
        return self._pad_top

    @property
    def pad_right(self) -> int:
        return self._pad_right

    @property
    def pad_front(self) -> int:
        return self._pad_front

    @property
    def pad_bottom(self) -> int:
        return self._pad_bottom

    @property
    def pad_left(self) -> int:
        return self._pad_left

    @property
    def pad_back(self) -> int:
        return self._pad_back

    @property
    def pool_type(self) -> str:
        return self._pool_type

    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> int:
        return self._coarse

    @property
    def coarse_out(self) -> int:
        return self._coarse

    @property
    def fine(self) -> int:
        if self.pool_type == "max":
            return self.kernel_rows * self.kernel_cols * self.kernel_depth
        else:
            return self._fine

    @kernel_rows.setter
    def kernel_rows(self, val: int) -> None:
        self._kernel_rows = val
        # self.update()

    @kernel_cols.setter
    def kernel_cols(self, val: int) -> None:
        self._kernel_cols = val
        # self.update()

    @kernel_depth.setter
    def kernel_depth(self, val: int) -> None:
        self._kernel_depth = val
        # self.update()

    @stride_rows.setter
    def stride_rows(self, val: int) -> None:
        self._stride_rows = val
        # self.update()

    @stride_cols.setter
    def stride_cols(self, val: int) -> None:
        self._stride_cols = val
        # self.update()

    @stride_depth.setter
    def stride_depth(self, val: int) -> None:
        self._stride_depth = val
        # self.update()

    @pad_top.setter
    def pad_top(self, val: int) -> None:
        self._pad_top = val
        # self.update()

    @pad_right.setter
    def pad_right(self, val: int) -> None:
        self._pad_right = val
        # self.update()

    @pad_front.setter
    def pad_front(self, val: int) -> None:
        self._pad_front = val
        # self.update()

    @pad_bottom.setter
    def pad_bottom(self, val: int) -> None:
        self._pad_bottom = val
        # self.update()

    @pad_left.setter
    def pad_left(self, val: int) -> None:
        self._pad_left = val
        # self.update()

    @pad_back.setter
    def pad_back(self, val: int) -> None:
        self._pad_back = val
        # self.update()

    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
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

    @fine.setter
    def fine(self, val: int) -> None:
        self._fine = val
        # self.update()

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

        if self.backend == "chisel":
            # pad
            self.modules['pad3d'].rows     = self.rows
            self.modules['pad3d'].cols     = self.cols
            self.modules['pad3d'].depth    = self.depth
            self.modules['pad3d'].channels = self.channels//self.coarse
            self.modules['pad3d'].data_width = self.data_t.width
            if self.data_packing:
                self.modules['pad3d'].streams = self.coarse
            self.modules['pad3d'].pad_top = self.pad_top
            self.modules['pad3d'].pad_bottom = self.pad_bottom
            self.modules['pad3d'].pad_left = self.pad_left
            self.modules['pad3d'].pad_right = self.pad_right
            self.modules['pad3d'].pad_front = self.pad_front
            self.modules['pad3d'].pad_back = self.pad_back

            # sliding window
            self.modules['sliding_window3d'].rows     = self.rows_in() + self.pad_top + self.pad_bottom
            self.modules['sliding_window3d'].cols     = self.cols_in() + self.pad_left + self.pad_right
            self.modules['sliding_window3d'].depth    = self.depth_in() + self.pad_front + self.pad_back
            self.modules['sliding_window3d'].channels = int(self.channels_in()/self.coarse)
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
        elif self.backend == "hls":
            self.modules['sliding_window3d'].rows     = self.rows_in()
            self.modules['sliding_window3d'].cols     = self.cols_in()
            self.modules['sliding_window3d'].depth    = self.depth_in()
            self.modules['sliding_window3d'].channels = int(self.channels_in()/self.coarse)
            self.modules['sliding_window3d'].data_width = self.data_t.width
            if self.data_packing:
                self.modules['sliding_window3d'].streams = self.coarse

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

        if self.backend == "chisel":
            pad_rsc     = self.modules['pad3d'].rsc()
            sw_rsc      = self.modules['sliding_window3d'].rsc()
            pool_rsc    = self.modules['pool3d'].rsc()

            # Total
            if self.data_packing:
                return {
                    "LUT"  :  pad_rsc['LUT'] + sw_rsc['LUT'] +
                            pool_rsc['LUT'],
                    "FF"   :  pad_rsc['FF'] + sw_rsc['FF'] +
                            pool_rsc['FF'],
                    "BRAM" :  pad_rsc['BRAM'] + sw_rsc['BRAM'] +
                            pool_rsc['BRAM'],
                    "DSP" :   pad_rsc['DSP'] + sw_rsc['DSP'] +
                            pool_rsc['DSP']
                }
            else:
                return {
                    "LUT"  :  pad_rsc['LUT']*self.coarse +
                            sw_rsc['LUT']*self.coarse +
                            pool_rsc['LUT']*self.coarse,
                    "FF"   :  pad_rsc['FF']*self.coarse +
                            sw_rsc['FF']*self.coarse +
                            pool_rsc['FF']*self.coarse,
                    "BRAM" :  pad_rsc['BRAM']*self.coarse +
                            sw_rsc['BRAM']*self.coarse +
                            pool_rsc['BRAM']*self.coarse,
                    "DSP" :   pad_rsc['DSP']*self.coarse +
                            sw_rsc['DSP']*self.coarse +
                            pool_rsc['DSP']*self.coarse
                }
        elif self.backend == "hls":
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
