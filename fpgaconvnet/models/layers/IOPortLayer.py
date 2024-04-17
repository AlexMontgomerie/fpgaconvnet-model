import pydot
import numpy as np

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.layers import Layer
from fpgaconvnet.models.modules import IOPort

class IOPortLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int,
            direction: str,
            connectivity: str = "off-chip",
            dma_stream_width: int = 128,
            dma_burst_size: int = 256,
            dma_clock_freq: int = 1067,
            data_t: FixedPoint = FixedPoint(16,8),
            backend: str = "chisel",
            regression_model: str = "linear_regression"
        ):

        # initialise parent class
        super().__init__(rows, cols, channels,
                coarse, coarse, data_t=data_t)

        # save parameters
        self._coarse = coarse
        self.direction = direction
        self.dma_stream_width = dma_stream_width
        self.dma_burst_size = dma_burst_size
        self.dma_clock_freq = dma_clock_freq
        self.connectivity = connectivity

        # direction
        assert direction in ["in", "out"], f"{direction} is an invalid direction"

        # connectivity type
        assert connectivity in ["on-chip", "off-chip"], f"{connectivity} is an invalid connectivity type"

        # backend flag
        assert backend in ["chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # regression model
        assert regression_model in ["linear_regression", "xgboost"], f"{regression_model} is an invalid regression model"
        self.regression_model = regression_model

        self.available_bw = self.mem_bw_in if self.direction == "in" else self.mem_bw_out

        # initialise modules
        self.modules["ioport"] = IOPort(self.rows, self.cols,
                self.channels, self._coarse, self.available_bw, self.direction, self.dma_stream_width, self.dma_burst_size, backend=self.backend, regression_model=self.regression_model)

        self.update()

    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> int:
        return self._coarse

    @property
    def coarse_out(self) -> int:
        return self._coarse

    def streams_in(self) -> int:
        return 1 if self.direction == "in" else self.coarse_in

    def streams_out(self) -> int:
        return 1 if self.direction == "out" else self.coarse_out

    def width_in(self) -> int:
        return self.dma_stream_width if self.direction == "in" else self.data_t.width

    def width_out(self) -> int:
        return self.dma_stream_width if self.direction == "out" else self.data_t.width

    def layer_info(self, parameters, batch_size=1):
        Layer.layer_info(self, parameters, batch_size)

    def update(self):
        self.modules["ioport"].rows = self.rows
        self.modules["ioport"].cols = self.cols
        self.modules["ioport"].channels = self.channels
        self.modules["ioport"].num_ports = self.coarse
        self.modules["ioport"].mem_bw  = self.mem_bw_in if self.direction == "in" else self.mem_bw_out
        self.modules["ioport"].dma_stream_width = self.dma_stream_width
        self.modules["ioport"].dma_burst_size = self.dma_burst_size
        self.modules["ioport"].data_width = self.data_t.width

    def resource(self):

        # get ioport resources
        ioport_rsc = self.modules['ioport'].rsc()

        # Total
        return {
            "LUT"  :  ioport_rsc['LUT'],
            "FF"   :  ioport_rsc['FF'],
            "BRAM" :  ioport_rsc['BRAM'],
            "DSP" :   ioport_rsc['DSP'],
        }

    def visualise(self,name):

        # create layer cluster
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="moccasin")

        # add ioport module
        ioport_name = "_".join([name,"ioport"])
        cluster.add_node(self.modules["ioport"].visualise(ioport_name))

        # get nodes in and out
        nodes_in  = [ ioport_name for i in range(self.streams_in()) ]
        nodes_out = [ ioport_name for i in range(self.streams_out()) ]

        # return module
        return cluster, nodes_in, nodes_out

    def functional_model(self, data, batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        # return output featuremap
        return data

