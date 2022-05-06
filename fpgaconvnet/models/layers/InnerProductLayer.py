import numpy as np
import math
import pydot
import torch

from fpgaconvnet.models.layers.utils import get_factors

from fpgaconvnet.tools.resource_model import bram_memory_resource_model

from fpgaconvnet.models.modules import SlidingWindow
from fpgaconvnet.models.modules import Conv
from fpgaconvnet.models.modules import Fork
from fpgaconvnet.models.modules import Accum
from fpgaconvnet.models.modules import Glue
from fpgaconvnet.models.modules import Bias
from fpgaconvnet.models.layers import Layer

class InnerProductLayer(Layer):
    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            input_width: int = 16,
            output_width: int = 16,
            weight_width: int = 16,
            acc_width: int = 16,
            biases_width: int = 16,
            has_bias: int = 0
        ):

        # initialise parent class
        super().__init__(rows, cols, channels, coarse_in,
                coarse_out, data_width=input_width)

        # save the widths
        self.input_width = input_width
        self.output_width = output_width
        self.weight_width = weight_width
        self.acc_width = acc_width
        self.biases_width = biases_width
        # save bias flag
        self.has_bias = has_bias

        # update flags
        # self.flags['channel_dependant'] = True
        # self.flags['transformable']     = True

        # save parameters
        self._filters = filters

        # init modules
        self.modules["fork"] = Fork(self.rows_in(), self.cols_in(), self.channels_in(), 1,
                self.coarse_out)
        self.modules["conv"] = Conv(1,1,self.channels_in()*self.rows_in()*self.cols_in(),
                self.filters, 1, 1, 1)
        self.modules["accum"] = Accum(1,1,self.channels_in()*self.rows_in()*self.cols_in(),
                self.filters, 1)
        self.modules["glue"] = Glue(1,1,self.channels_in()*self.rows_in()*self.cols_in(),
                self.filters, self.coarse_in, self.coarse_out)
        self.modules["bias"] = Bias(1,1,self.channels_in()*self.rows_in()*self.cols_in(),
                self.filters)

        self.update()

    @property
    def filters(self) -> int:
        return self._filters

    @filters.setter
    def filters(self, val: int) -> None:
        self._filters = val
        self.update()

    def rows_out(self) -> int:
        return 1

    def cols_out(self) -> int:
        return 1

    def channels_out(self) -> int:
        return self.filters

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.filters      = self.filters
        parameters.input_width  = self.input_width
        parameters.output_width = self.output_width
        parameters.weight_width = self.weight_width
        parameters.acc_width    = self.acc_width
        parameters.biases_width = self.biases_width
        parameters.has_bias     = self.has_bias

    def update(self): # TODO: update all parameters
        # fork
        self.modules['fork'].rows     = self.rows_in()
        self.modules['fork'].cols     = self.cols_in()
        self.modules['fork'].channels = self.channels_in()//self.coarse_in
        self.modules['fork'].coarse   = self.coarse_out
        self.modules['fork'].data_width = self.input_width
        # conv
        self.modules['conv'].rows     = 1
        self.modules['conv'].cols     = 1
        self.modules['conv'].channels =\
                                self.rows_in()*self.cols_in()*self.channels_in()//self.coarse_in
        self.modules['conv'].filters  = self.filters//self.coarse_out
        self.modules['conv'].fine     = 1
        self.modules['conv'].data_width = self.input_width
        self.modules['conv'].weight_width = self.weight_width
        self.modules['conv'].acc_width = self.acc_width
        # accum
        self.modules['accum'].rows     = 1
        self.modules['accum'].cols     = 1
        self.modules['accum'].channels =\
                                self.rows_in()*self.cols_in()*self.channels_in()//self.coarse_in
        self.modules['accum'].filters  = self.filters//self.coarse_out
        self.modules['accum'].data_width = self.acc_width
        # glue
        self.modules['glue'].rows = 1
        self.modules['glue'].cols = 1
        self.modules['glue'].filters    = self.filters
        self.modules['glue'].coarse_in  = self.coarse_in
        self.modules['glue'].coarse_out = self.coarse_out
        self.modules['glue'].data_width = self.output_width
        self.modules['glue'].acc_width  = self.acc_width
        # bias FIXME
        self.modules['bias'].rows           = 1#self.rows_out()
        self.modules['bias'].cols           = 1#self.cols_out()
        self.modules['bias'].filters        = self.filters
        self.modules['bias'].data_width     = self.output_width
        self.modules['bias'].biases_width   = self.biases_width

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

        fork_rsc    = self.modules['fork'].rsc()
        conv_rsc    = self.modules['conv'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        if self.rows_in()*self.cols_in()*self.channels_in()//self.coarse_in == 1:
            accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        glue_rsc    = self.modules['glue'].rsc()
        if self.coarse_in == 1:
            glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        bias_rsc    = self.modules['bias'].rsc()
        # condition if there are no biases for the layer
        if self.has_bias:
            bias_rsc = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        # TODO: add to modules instead
        weights_memory_depth = float(self.filters*self.channels_in()*self.rows_in()*\
                self.cols_in())/float(self.coarse_in*self.coarse_out)
        weights_bram_usage = \
            bram_memory_resource_model(int(weights_memory_depth), self.weight_width)*\
            self.coarse_in*self.coarse_out

        # FIXME: sort mem requirements correctly
        bias_memory_depth = float(self.filters*self.rows_in()*\
            self.cols_in())/float(self.coarse_out)
        biases_bram_usage = \
            bram_memory_resource_model(int(bias_memory_depth), self.biases_width)*self.coarse_out
        # Total
        return {
            "LUT"  :  fork_rsc['LUT']*self.coarse_in +
                      conv_rsc['LUT']*self.coarse_in*self.coarse_out +
                      accum_rsc['LUT']*self.coarse_in*self.coarse_out +
                      glue_rsc['LUT'] +
                      bias_rsc['LUT']*self.coarse_out,
            "FF"   :  fork_rsc['FF']*self.coarse_in +
                      conv_rsc['FF']*self.coarse_in*self.coarse_out +
                      accum_rsc['FF']*self.coarse_in*self.coarse_out +
                      glue_rsc['FF'] +
                      bias_rsc['FF']*self.coarse_out,
            "BRAM" :  fork_rsc['BRAM']*self.coarse_in +
                      conv_rsc['BRAM']*self.coarse_in*self.coarse_out +
                      accum_rsc['BRAM']*self.coarse_out +
                      glue_rsc['BRAM'] +
                      weights_bram_usage +
                      bias_rsc['BRAM']*self.coarse_out +
                      biases_bram_usage,
            "DSP"  :  fork_rsc['DSP']*self.coarse_in +
                      conv_rsc['DSP']*self.coarse_in*self.coarse_out +
                      accum_rsc['DSP']*self.coarse_out +
                      glue_rsc['DSP'] +
                      bias_rsc['DSP']*self.coarse_out
        }

    def visualise(self, name):

        cluster = pydot.Cluster(name, label=name,
                style="dashed", bgcolor="lightyellow")

        # names
        fork_name = [""]*self.coarse_in
        conv_name = [[""]*self.coarse_in]*self.coarse_out
        accum_name = [[""]*self.coarse_in]*self.coarse_out
        glue_name = [""]*self.coarse_out
        bias_name = [""]*self.coarse_out

        for i in range(self.coarse_in):
            # define names
            fork_name[i] = "_".join([name, "fork", str(i)])
            # add nodes
            cluster.add_node(self.modules["fork"].visualise(fork_name[i]))

            # iterate over coarse out
            for j in range(self.coarse_out):
                # define names
                conv_name[j][i] = "_".join([name, "conv", str(j), str(i)])
                accum_name[j][i] = "_".join([name, "accum", str(j), str(i)])
                glue_name[j] = "_".join([name, "glue", str(j)])
                bias_name[j] = "_".join([name, "bias", str(j)])

                # add nodes
                cluster.add_node(self.modules["conv"].visualise(conv_name[j][i]))
                cluster.add_node(self.modules["accum"].visualise(accum_name[j][i]))
                cluster.add_node(self.modules["glue"].visualise(glue_name[j]))
                cluster.add_node(self.modules["bias"].visualise(bias_name[j]))

                # add edges
                cluster.add_edge(pydot.Edge(fork_name[i], conv_name[j][i]))
                cluster.add_edge(pydot.Edge(conv_name[j][i], accum_name[j][i]))
                cluster.add_edge(pydot.Edge(accum_name[j][i], glue_name[j]))
                cluster.add_edge(pydot.Edge(glue_name[j], bias_name[j]))

        return cluster, fork_name, bias_name

    def functional_model(self,data,weights,bias,batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters ,   "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.rows_in()*self.cols_in()*self.channels_in(),\
                                                    "ERROR (weights): invalid channel dimension"


        # instantiate inner product layer
        inner_product_layer = torch.nn.Linear(
                self.channels_in()*self.rows_in()*self.cols_in(), self.filters)#, bias=False)

        # update weights
        inner_product_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        inner_product_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # return output featuremap
        data = np.moveaxis(data, -1, 0).flatten()
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return inner_product_layer(torch.from_numpy(data)).detach().numpy()

