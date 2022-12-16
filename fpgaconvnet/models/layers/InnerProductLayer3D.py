import numpy as np
import math
import pydot
import torch

from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.tools.resource_analytical_model import bram_memory_resource_model
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.layers import Layer3D

from fpgaconvnet.models.modules import Conv3D
from fpgaconvnet.models.modules import Fork3D
from fpgaconvnet.models.modules import Accum3D
from fpgaconvnet.models.modules import Glue3D
from fpgaconvnet.models.modules import Bias3D
from fpgaconvnet.models.modules import VectorDot3D

class InnerProductLayer3D(Layer3D):
    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            depth: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            input_t: FixedPoint = FixedPoint(16,8),
            output_t: FixedPoint = FixedPoint(16,8),
            weight_t: FixedPoint = FixedPoint(16,8),
            acc_t: FixedPoint = FixedPoint(32,16),
            has_bias: int = 0,
            backend: str = "chisel",
            double_buffered: bool = True,
            stream_weights: bool = False,

        ):

        # initialise parent class
        super().__init__(rows, cols, depth, channels, coarse_in,
                coarse_out, data_t=input_t)

        # save data types
        self.input_t = input_t
        self.output_t = output_t
        self.weight_t = weight_t
        self.acc_t = acc_t

        # save bias flag
        self.has_bias = has_bias

        # save parameters
        self._filters = filters

        # weights buffering flag
        self.double_buffered = double_buffered
        self.stream_weights = stream_weights

        # backend flag
        assert backend in ["hls", "chisel"], f"{backend} is an invalid backend"
        self.backend = backend

        # init modules
        self.modules["fork3d"] = Fork3D(self.rows_in(), self.cols_in(), self.depth_in(), self.channels_in(), 1, 1, 1, self.coarse_out, backend=self.backend)
        if self.backend == "hls":
            self.modules["conv3d"] = Conv3D(1,1,1,
                    self.channels_in()*self.rows_in()*self.cols_in()*self.depth_in(),
                    self.filters, 1, 1, 1, backend=self.backend)
        elif self.backend == "chisel":
            self.modules["vector_dot3d"] = VectorDot3D(1, 1, 1,
                    self.channels_in()*self.rows_in()*self.cols_in()*self.depth_in(),
                    self.filters, 1, backend=self.backend)
        self.modules["accum3d"] = Accum3D(1,1,1,self.channels_in()*self.rows_in()*self.cols_in()*self.depth_in(),
                self.filters, 1, backend=self.backend)
        self.modules["glue3d"] = Glue3D(1,1,1,self.channels_in()*self.rows_in()*self.cols_in()*self.depth_in(),
                self.filters, self.coarse_in, self.coarse_out, backend=self.backend)
        self.modules["bias3d"] = Bias3D(1,1,1,self.channels_in()*self.rows_in()*self.cols_in()*self.depth_in(),
                self.filters, backend=self.backend)

        self.update()

    @property
    def filters(self) -> int:
        return self._filters

    @filters.setter
    def filters(self, val: int) -> None:
        self._filters = val
        # self.update()

    def rows_out(self) -> int:
        return 1

    def cols_out(self) -> int:
        return 1

    def depth_out(self) -> int:
        return 1

    def channels_out(self) -> int:
        return self.filters

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)
        parameters.filters  = self.filters
        parameters.has_bias = self.has_bias
        self.input_t.to_protobuf(parameters.input_t)
        self.output_t.to_protobuf(parameters.output_t)
        self.weight_t.to_protobuf(parameters.weight_t)
        self.acc_t.to_protobuf(parameters.acc_t)
        parameters.data_t.Clear()

    def update(self):
        # fork
        self.modules['fork3d'].rows     = self.rows_in()
        self.modules['fork3d'].cols     = self.cols_in()
        self.modules['fork3d'].depth    = self.depth_in()
        self.modules['fork3d'].channels = self.channels_in()//self.coarse_in
        self.modules['fork3d'].coarse   = self.coarse_out
        self.modules['fork3d'].data_width = self.input_t.width
        if self.backend == "hls":
            # conv
            self.modules['conv3d'].rows     = 1
            self.modules['conv3d'].cols     = 1
            self.modules['conv3d'].depth    = 1
            self.modules['conv3d'].channels =\
                self.rows_in()*self.cols_in()*self.depth_in()*self.channels_in()//self.coarse_in
            self.modules['conv3d'].filters  = self.filters//self.coarse_out
            self.modules['conv3d'].fine     = 1
            self.modules['conv3d'].data_width = self.input_t.width
            self.modules['conv3d'].weight_width = self.weight_t.width
            self.modules['conv3d'].acc_width = self.acc_t.width
        elif self.backend == "chisel":
            # vector dot
            self.modules['vector_dot3d'].rows     = self.rows_out()
            self.modules['vector_dot3d'].cols     = self.cols_out()
            self.modules['vector_dot3d'].depth    = self.depth_out()
            self.modules['vector_dot3d'].channels =\
                self.rows_in()*self.cols_in()*self.depth_in()*self.channels_in()//self.coarse_in
            self.modules['vector_dot3d'].filters  = self.filters//self.coarse_out
            self.modules['vector_dot3d'].fine     = 1
            self.modules['vector_dot3d'].data_width     = self.input_t.width
            self.modules['vector_dot3d'].weight_width   = self.weight_t.width
            self.modules['vector_dot3d'].acc_width      = self.acc_t.width
        # accum
        self.modules['accum3d'].rows     = 1
        self.modules['accum3d'].cols     = 1
        self.modules['accum3d'].depth    = 1
        self.modules['accum3d'].channels =\
            self.rows_in()*self.cols_in()*self.depth_in()*self.channels_in()//self.coarse_in
        self.modules['accum3d'].filters  = self.filters//self.coarse_out
        self.modules['accum3d'].data_width = self.acc_t.width
        # glue
        self.modules['glue3d'].rows = 1
        self.modules['glue3d'].cols = 1
        self.modules['glue3d'].depth = 1
        self.modules['glue3d'].filters    = self.filters
        self.modules['glue3d'].coarse_in  = self.coarse_in
        self.modules['glue3d'].coarse_out = self.coarse_out
        self.modules['glue3d'].data_width = self.acc_t.width
        # bias
        self.modules['bias3d'].rows           = 1 #self.rows_out()
        self.modules['bias3d'].cols           = 1 #self.cols_out()
        self.modules['bias3d'].depth          = 1 #self.depth_out()
        self.modules['bias3d'].filters        = self.filters

    def get_weights_reloading_feasible(self):
        return get_factors(int(self.filters/self.coarse_out))

    def get_parameters_size(self):
        weights_size = self.channels * self.filters
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def resource(self):

        # get module resource models
        fork_rsc        = self.modules['fork3d'].rsc()
        if self.backend == "chisel":
            vector_dot_rsc  = self.modules['vector_dot3d'].rsc()
        elif self.backend == "hls":
            conv_rsc        = self.modules['conv3d'].rsc()
        accum_rsc       = self.modules['accum3d'].rsc()
        glue_rsc        = self.modules['glue3d'].rsc()
        bias_rsc        = self.modules['bias3d'].rsc()

        # remove redundant modules
        if self.coarse_out == 1:
            fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.rows_in()*self.cols_in()*self.depth_in()*self.channels_in()//self.coarse_in == 1:
            accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_in == 1:
            glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.has_bias:
            bias_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        if self.backend == "chisel":
            # accumulate resource usage based on coarse factors
            rsc = { rsc_type: (
                fork_rsc[rsc_type]*self.coarse_in +
                vector_dot_rsc[rsc_type]*self.coarse_in*self.coarse_out +
                accum_rsc[rsc_type]*self.coarse_in*self.coarse_out +
                glue_rsc[rsc_type]*self.coarse_out +
                bias_rsc[rsc_type]*self.coarse_out
            ) for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

        elif self.backend == "hls":

            # accumulate resource usage based on coarse factors
            rsc = { rsc_type: (
                fork_rsc[rsc_type]*self.coarse_in +
                conv_rsc[rsc_type]*self.coarse_in*self.coarse_out +
                accum_rsc[rsc_type]*self.coarse_in*self.coarse_out +
                glue_rsc[rsc_type]*self.coarse_out +
                bias_rsc[rsc_type]*self.coarse_out
            ) for rsc_type in ["LUT", "FF", "DSP", "BRAM"] }

        # weight usage
        weights_memory_depth = float(self.filters*self.channels_in()*self.rows_in()*\
                self.cols_in()*self.depth_in())/float(self.coarse_in*self.coarse_out)

        if self.double_buffered:
            weights_memory_depth *= 2

        weights_bram_usage = bram_memory_resource_model(
                    int(weights_memory_depth), self.weight_t.width) * self.coarse_in * self.coarse_out

        if self.stream_weights:
            weights_bram_usage = 0

        # bias usage FIXME depth, FIXME bram usage
        bias_memory_depth = float(self.filters) / float(self.coarse_out)
        biases_bram_usage = bram_memory_resource_model(
                    int(bias_memory_depth),self.acc_t.width) * self.coarse_out

        # add weights and bias to resources
        rsc["BRAM"] += weights_bram_usage + biases_bram_usage

        # return total resource
        return rsc

    def visualise(self, name):

        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightyellow")

        # names
        fork_name = [""]*self.coarse_in
        vector_dot_name = [[""]*self.coarse_in]*self.coarse_out
        accum_name = [[""]*self.coarse_in]*self.coarse_out
        glue_name = [""]*self.coarse_out
        bias_name = [""]*self.coarse_out

        for i in range(self.coarse_in):
            # define names
            fork_name[i] = "_".join([name, "fork3d", str(i)])
            # add nodes
            cluster.add_node(self.modules["fork3d"].visualise(fork_name[i]))

            # iterate over coarse out
            for j in range(self.coarse_out):
                # define names
                vector_dot_name[j][i] = "_".join([name, "vector_dot3d", str(j), str(i)])
                accum_name[j][i] = "_".join([name, "accum3d", str(j), str(i)])
                glue_name[j] = "_".join([name, "glue3d", str(j)])
                bias_name[j] = "_".join([name, "bias3d", str(j)])

                # add nodes
                cluster.add_node(self.modules["vector_dot3d"].visualise(vector_dot_name[j][i]))
                cluster.add_node(self.modules["accum3d"].visualise(accum_name[j][i]))
                cluster.add_node(self.modules["glue3d"].visualise(glue_name[j]))
                cluster.add_node(self.modules["bias3d"].visualise(bias_name[j]))

                # add edges
                cluster.add_edge(pydot.Edge(fork_name[i], vector_dot_name[j][i]))
                cluster.add_edge(pydot.Edge(vector_dot_name[j][i], accum_name[j][i]))
                cluster.add_edge(pydot.Edge(accum_name[j][i], glue_name[j]))
                cluster.add_edge(pydot.Edge(glue_name[j], bias_name[j]))

        return cluster, fork_name, bias_name

    def functional_model(self,data,weights,bias,batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.depth_in()   , "ERROR (data): invalid depth dimension"
        assert data.shape[3] == self.channels_in(), "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters ,   "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.rows_in()*self.cols_in()*self.depth_in()*self.channels_in(),\
                                                    "ERROR (weights): invalid channel dimension"


        # instantiate inner product layer
        inner_product_layer = torch.nn.Linear(
                self.channels_in()*self.rows_in()*self.cols_in()*self.depth_in(), self.filters)#, bias=False)

        # update weights
        inner_product_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        inner_product_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # return output featuremap
        data = np.moveaxis(data, [-1, -2], [0, 1]).flatten()
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return inner_product_layer(torch.from_numpy(data)).detach().numpy()

