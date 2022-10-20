import pydot
import numpy as np

from fpgaconvnet.data_types import FixedPoint

from fpgaconvnet.models.layers import Layer
from fpgaconvnet.models.modules import Squeeze

class SqueezeLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
            data_t: FixedPoint = FixedPoint(16,8),
        ):

        # initialise parent class
        super().__init__(rows, cols, channels,
                coarse_in, coarse_out,data_t=data_t)

        # initialise modules
        self.modules["squeeze"] = Squeeze(self.rows, self.cols,
                self.channels, self.coarse_in, self.coarse_out)

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)

    def update(self):
        self.modules["squeeze"].rows = self.rows
        self.modules["squeeze"].cols = self.cols
        self.modules["squeeze"].channels = self.channels
        self.modules["squeeze"].coarse_in = self.coarse_in
        self.modules["squeeze"].coarse_out = self.coarse_out

    def visualise(self,name):

        # create layer cluster
        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="moccasin")

        # add squeeze module
        squeeze_name = "_".join([name,"squeeze"])
        cluster.add_node(self.modules["squeeze"].visualise(squeeze_name))

        # get nodes in and out
        nodes_in  = [ squeeze_name for i in range(self.streams_in()) ]
        nodes_out = [ squeeze_name for i in range(self.streams_out()) ]

        # return module
        return cluster, nodes_in, nodes_out

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        return np.repeat(data[np.newaxis,...], batch_size, axis=0)

