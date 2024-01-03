import math
from typing import List, Union

import numpy as np
import pydot

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import SlidingWindow
from fpgaconvnet.models.modules import Pool
from fpgaconvnet.models.modules import Pad
from fpgaconvnet.models.layers import Layer

class PoolingLayer(Layer):

    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int = 1,
            pool_type: str = 'max',
            kernel_rows: int = 1,
            kernel_cols: int = 1,
            stride_rows: int = 1,
            stride_cols: int = 1,
            pad_top: int = 0,
            pad_right: int = 0,
            pad_bottom: int = 0,
            pad_left: int = 0,
            fine: int = 1,
            data_t: FixedPoint = FixedPoint(16,8),
            backend: str = "chisel",
            regression_model: str = "linear_regression",
            input_compression_ratio: list = [1.0],
            output_compression_ratio: list = [1.0]
        ):

        # initialise parent class
        super().__init__(rows, cols, channels,
                coarse, coarse, data_t=data_t,
                input_compression_ratio=input_compression_ratio,
                output_compression_ratio=output_compression_ratio)

        # update flags
        # self.flags['transformable'] = True

        # update parameters
        self._kernel_rows = kernel_rows
        self._kernel_cols = kernel_cols
        self._stride_rows = stride_rows
        self._stride_cols = stride_cols
        self._pad_top       = pad_top
        self._pad_right     = pad_right
        self._pad_bottom    = pad_bottom
        self._pad_left      = pad_left
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
            self.modules["pad"] = Pad(
                    self.rows_in(), self.cols_in(), self.channels_in()//self.coarse,
                    self.pad_top, self.pad_bottom, self.pad_left, self.pad_right, backend=self.backend,
                    regression_model=self.regression_model)

            self.modules["sliding_window"] = SlidingWindow(
                self.rows_in() + self.pad_top + self.pad_bottom,
                self.cols_in() + self.pad_left + self.pad_right,
                self.channels_in()//self.coarse,
                self.kernel_size,
                self.stride,
                0, 0, 0, 0, backend=self.backend,
                regression_model=self.regression_model)
        elif self.backend == "hls":
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

    def start_depth(self):
        # return (self.kernel_rows-1)*self.cols*self.channels//self.streams_in() + \
        #     (self.kernel_cols-1)*self.channels//self.streams_in()
        # return (self.kernel_rows-1)*self.cols*self.channels + \
        #         (self.kernel_cols-1)*self.channels - \
        #         ( self.pad_top * self.cols * self.channels + \
        #         (self.pad_left+self.pad_right)*self.channels ) + \
        #         self.channels
        return (self.kernel_rows-1-self.pad_top)*self.cols*self.channels//self.streams_in() + \
               (self.kernel_cols-1-self.pad_left)*self.channels//self.streams_in() + \
                self.channels//self.streams_in()

    def get_operations(self):
        return self.channels_in()*self.rows_out()*self.cols_out()\
            *self.kernel_size[0]*self.kernel_size[1]

    def rows_out(self) -> int:
        return self.modules["sliding_window"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window"].cols_out()

    @property
    def kernel_size(self) -> List[int]:
        return [ self._kernel_rows, self._kernel_cols ]

    @property
    def kernel_rows(self) -> int:
        return self._kernel_rows

    @property
    def kernel_cols(self) -> int:
        return self._kernel_cols

    @property
    def stride(self) -> List[int]:
        return [ self._stride_rows, self._stride_cols ]

    @property
    def stride_rows(self) -> int:
        return self._stride_rows

    @property
    def stride_cols(self) -> int:
        return self._stride_cols

    @property
    def pad(self) -> List[int]:
        return [
            self._pad_top,
            self._pad_left,
            self._pad_bottom,
            self._pad_right,
        ]

    @property
    def pad_top(self) -> int:
        return self._pad_top

    @property
    def pad_right(self) -> int:
        return self._pad_right

    @property
    def pad_bottom(self) -> int:
        return self._pad_bottom

    @property
    def pad_left(self) -> int:
        return self._pad_left

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
            return self.kernel_size[0] * self.kernel_size[1]
        else:
            return self._fine

    @kernel_size.setter
    def kernel_size(self, val: List[int]) -> None:
        self._kernel_rows = val[0]
        self._kernel_cols = val[1]
        # self.update()

    @kernel_rows.setter
    def kernel_rows(self, val: int) -> None:
        self._kernel_rows = val
        # self.update()

    @kernel_cols.setter
    def kernel_cols(self, val: int) -> None:
        self._kernel_cols = val
        # self.update()

    @stride.setter
    def stride(self, val: List[int]) -> None:
        self._stride_rows = val[0]
        self._stride_cols = val[1]
        # self.update()

    @stride_rows.setter
    def stride_rows(self, val: int) -> None:
        self._stride_rows = val
        # self.update()

    @stride_cols.setter
    def stride_cols(self, val: int) -> None:
        self._stride_cols = val
        # self.update()

    @pad.setter
    def pad(self, val: List[int]) -> None:
        self._pad_top    = val[0]
        self._pad_right  = val[3]
        self._pad_bottom = val[2]
        self._pad_left   = val[1]
        # self.update()

    @pad_top.setter
    def pad_top(self, val: int) -> None:
        self._pad_top = val
        # self.update()

    @pad_right.setter
    def pad_right(self, val: int) -> None:
        self._pad_right = val
        # self.update()

    @pad_bottom.setter
    def pad_bottom(self, val: int) -> None:
        self._pad_bottom = val
        # self.update()

    @pad_left.setter
    def pad_left(self, val: int) -> None:
        self._pad_left = val
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
        if self.backend == "chisel":
            # pad
            self.modules['pad'].rows     = self.rows
            self.modules['pad'].cols     = self.cols
            self.modules['pad'].channels = self.channels//self.coarse
            self.modules['pad'].data_width = self.data_t.width
            if self.data_packing:
                self.modules['pad'].streams = self.coarse
            self.modules['pad'].pad_top = self.pad_top
            self.modules['pad'].pad_bottom = self.pad_bottom
            self.modules['pad'].pad_left = self.pad_left
            self.modules['pad'].pad_right = self.pad_right

            # sliding window
            self.modules['sliding_window'].rows     = self.rows_in() + self.pad_top + self.pad_bottom
            self.modules['sliding_window'].cols     = self.cols_in() + self.pad_left + self.pad_right
            self.modules['sliding_window'].channels = int(self.channels_in()/self.coarse)
            self.modules['sliding_window'].kernel_cols = self.kernel_cols
            self.modules['sliding_window'].kernel_rows = self.kernel_rows
            self.modules['sliding_window'].stride_cols = self.stride_cols
            self.modules['sliding_window'].stride_rows = self.stride_rows
            self.modules['sliding_window'].data_width = self.data_t.width
            if self.data_packing:
                self.modules['sliding_window'].streams = self.coarse
            self.modules['sliding_window'].pad_top = 0
            self.modules['sliding_window'].pad_bottom = 0
            self.modules['sliding_window'].pad_left = 0
            self.modules['sliding_window'].pad_right = 0
        elif self.backend == "hls":
            self.modules['sliding_window'].rows     = self.rows_in()
            self.modules['sliding_window'].cols     = self.cols_in()
            self.modules['sliding_window'].channels = int(self.channels_in()/self.coarse)
            self.modules['sliding_window'].data_width = self.data_t.width
            if self.data_packing:
                self.modules['sliding_window'].streams = self.coarse

        # pool
        self.modules['pool'].rows     = self.rows_out()
        self.modules['pool'].cols     = self.cols_out()
        self.modules['pool'].channels = int(self.channels_in()/self.coarse)
        self.modules['pool'].data_width = self.data_t.width
        if self.data_packing:
            self.modules['pool'].streams = self.coarse

    def get_fine_feasible(self):
        return [1]

    def resource(self):

        if self.backend == "chisel":
            pad_rsc     = self.modules['pad'].rsc()
            sw_rsc      = self.modules['sliding_window'].rsc()
            pool_rsc    = self.modules['pool'].rsc()

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
        if self.pool_type == 'max':
            pooling_layer = torch.nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=self.pad[0])
        elif self.pool_type == 'avg':
            pooling_layer = torch.nn.AvgPool2d(self.kernel_size, stride=self.stride, padding=self.pad[0])

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return pooling_layer(torch.from_numpy(data)).detach().numpy()
