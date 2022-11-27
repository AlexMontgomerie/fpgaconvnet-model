import math
from typing import Union, List

import torch
import numpy as np
import pydot

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.modules import SlidingWindow
from fpgaconvnet.models.modules import Pool
from fpgaconvnet.models.modules import MaxPool
from fpgaconvnet.models.layers import Layer

class PoolingLayer(Layer):

    def format_kernel_size(self, kernel_size):
        if isinstance(kernel_size, int):
            return [kernel_size, kernel_size]
        elif isinstance(kernel_size, list):
            assert len(kernel_size) == 2, "Must specify two kernel dimensions"
            return kernel_size
        else:
            raise TypeError

    def format_stride(self, stride):
        if isinstance(stride, int):
            return [stride, stride]
        elif isinstance(stride, list):
            assert len(stride) == 2, "Must specify two stride dimensions"
            return stride
        else:
            raise TypeError

    def format_pad(self, pad):
        if isinstance(pad, int):
            return [
                    pad - (self.rows_in() - self.kernel_size[0] + 2*pad) % self.stride[0],
                    pad,
                    pad,
                    pad - (self.cols_in() - self.kernel_size[1] + 2*pad) % self.stride[1],
                ]
        elif isinstance(pad, list):
            assert len(pad) == 4, "Must specify four pad dimensions"
            return pad
        else:
            raise TypeError

    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int = 1,
            pool_type   ='max',
            kernel_size: Union[List[int], int] = 2,
            stride: Union[List[int], int] = 2,
            pad: Union[List[int], int] = 0,
            fine: int = 1,
            data_t: FixedPoint = FixedPoint(16,8),
            backend: str = "chisel"
        ):

        # initialise parent class
        super().__init__(rows, cols, channels,
                coarse, coarse, data_t=data_t)

        # update flags
        # self.flags['transformable'] = True

        # update parameters
        self._kernel_size = self.format_kernel_size(kernel_size)
        self._stride = self.format_stride(stride)
        self._pad = self.format_pad(pad)
        self._pool_type = pool_type
        self._coarse = coarse
        self._fine = fine

        self._pad_top = self._pad[0]
        self._pad_right = self._pad[3]
        self._pad_bottom = self._pad[2]
        self._pad_left = self._pad[1]

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # init modules
        self.modules["sliding_window"] = SlidingWindow(self.rows_in(),
                self.cols_in(), self.channels_in()//self.coarse,
                self.kernel_size, self.stride, self.pad_top,
                self.pad_right, self.pad_bottom, self.pad_left, backend=self.backend)
        self.modules["pool"] = MaxPool(self.rows_out(), self.cols_out(),
                self.channels_out()//self.coarse, kernel_size, backend=self.backend)

        self.update()

    def rows_out(self) -> int:
        return self.modules["sliding_window"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window"].cols_out()

    @property
    def kernel_size(self) -> List[int]:
        return self._kernel_size

    @property
    def stride(self) -> List[int]:
        return self._stride

    @property
    def pad(self) -> List[int]:
        return self._pad

    @property
    def pad_top(self) -> int:
        return self._pad[0]

    @property
    def pad_right(self) -> int:
        return self._pad[3]

    @property
    def pad_bottom(self) -> int:
        return self._pad[2]

    @property
    def pad_left(self) -> int:
        return self._pad[1]

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
        self._kernel_size = self.format_kernel_size(val)
        self.update()

    @stride.setter
    def stride(self, val: List[int]) -> None:
        self._stride = self.format_stride(val)
        self.update()

    @pad.setter
    def pad(self, val: List[int]) -> None:
        self._pad = self.format_pad(val)
        self.pad_top = self._pad[0]
        self.pad_right = self._pad[3]
        self.pad_bottom = self._pad[2]
        self.pad_left = self._pad[1]
        self.update()

    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    @fine.setter
    def fine(self, val: int) -> None:
        self._fine = val
        self.update()

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.kernel_size.extend([self.kernel_size[0], self.kernel_size[1]])
        parameters.stride.extend([self.stride[0], self.stride[1]])
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left

    def update(self):
        # sliding window
        self.modules['sliding_window'].rows     = self.rows_in()
        self.modules['sliding_window'].cols     = self.cols_in()
        self.modules['sliding_window'].channels = int(self.channels_in()/self.coarse)
        self.modules['sliding_window'].data_width = self.data_t.width
        # pool
        self.modules['pool'].rows     = self.rows_out()
        self.modules['pool'].cols     = self.cols_out()
        self.modules['pool'].channels = int(self.channels_in()/self.coarse)
        self.modules['pool'].data_width = self.data_t.width

    def get_fine_feasible(self):
        return [1]

    def resource(self):

        sw_rsc      = self.modules['sliding_window'].rsc()
        pool_rsc    = self.modules['pool'].rsc()

        # Total
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

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # instantiate pooling layer
        pooling_layer = torch.nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=self.pad[0])

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return pooling_layer(torch.from_numpy(data)).detach().numpy()
