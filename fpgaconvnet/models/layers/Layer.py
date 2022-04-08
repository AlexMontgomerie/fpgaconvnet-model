"""

"""

import pydot
import collections
from google.protobuf.json_format import MessageToDict
import numpy as np
from dataclasses import dataclass, field

from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.models.layers.utils import balance_module_rates

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.tools.resource_model import bram_stream_resource_model

@dataclass
class Layer:
    """
    Base class for all layer models.

    Attributes
    ----------
    buffer_depth: int, default: 0
        depth of incoming fifo buffers for each stream in.
    rows: int
        row dimension of input featuremap
    cols: int
        column dimension of input featuremap
    channels: int
        channel dimension of input featuremap
    coarse_in: int
        number of parallel streams per port into the layer.
    coarse_out: int
        number of parallel streams per port out of the layer.
    data_width: int
        bitwidth of featuremap pixels
    modules: dict
        dictionary of `fpgaconvnet_optimiser.models.Module`
        instances that make up the layer. These modules are
        used for the resource and performance models of the
        layer.
    """

    _rows: int
    _cols: int
    _channels: int
    _coarse_in: int
    _coarse_out: int
    data_width: int = field(default=16, init=True)
    buffer_depth: int = field(default=0, init=False)
    modules: dict = field(default_factory=collections.OrderedDict, init=False)

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def coarse_in(self) -> int:
        return self._coarse_in

    @property
    def coarse_out(self) -> int:
        return self._coarse_out

    @rows.setter
    def rows(self, val: int) -> None:
        self._rows = val
        self.update()

    @cols.setter
    def cols(self, val: int) -> None:
        self._cols = val
        self.update()

    @channels.setter
    def channels(self, val: int) -> None:
        self._channels = val
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        assert(val in self.get_coarse_in_feasible())
        self._coarse_in = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        assert(val in self.get_coarse_out_feasible())
        self._coarse_out = val
        self.update()

    def rows_in(self) -> int:
        """
        Returns
        -------
        int
            row dimension of the input featuremap
        """
        return self.rows

    def cols_in(self) -> int:
        """
        Returns
        -------
        int
            column dimension of the input featuremap
        """
        return self.cols

    def channels_in(self) -> int:
        """
        Returns
        -------
        int
            channel dimension of the input featuremap
        """
        return self.channels

    def rows_out(self) -> int:
        """
        Returns
        -------
        int
            row dimension of the output featuremap
        """
        return self.rows

    def cols_out(self) -> int:
        """
        Returns
        -------
        int
            column dimension of the output featuremap
        """
        return self.cols

    def channels_out(self) -> int:
        """
        Returns
        -------
        int
            channel dimension of the output featuremap
        """
        return self.channels

    def build_rates_graph(self):

        # create the rates graph
        rates_graph = np.zeros(shape=(len(self.modules.keys()),
                                      len(self.modules.keys())+1) , dtype=float )

        # iterate over modules
        for i, module in enumerate(self.modules.keys()):
            # update rates_graph
            rates_graph[i,i] = self.modules[module].rate_in()
            rates_graph[i,i+1] = self.modules[module].rate_out()

        # return rates_graph
        return rates_graph

    def rate_in(self) -> float:
        """
        Returns
        -------
        float
            rate of words into layer. As a fraction of a
            clock cycle.

            default is 1.0
        """
        return abs(balance_module_rates(self.build_rates_graph())[0,0])

    def rate_out(self) -> float:
        """
        Returns
        -------
        float
            rate of words out of the layer. As a fraction
            of a clock cycle.

            default is 1.0
        """
        return abs(balance_module_rates(
            self.build_rates_graph())[len(self.modules.keys())-1,len(self.modules.keys())])

    def streams_in(self) -> int:
        """
        Returns
        -------
        int
            number of parallel streams into the layer.
        """
        return self.coarse_in

    def streams_out(self) -> int:
        """
        Returns
        -------
        int
            number of parallel streams out of the layer.
        """
        return self.coarse_out

    def workload_in(self) -> int:
        """
        Returns
        -------
        int
            workload into layer from port `index` for a single
            featuremap. This is calculated by
            `rows_in()*cols_in()*channels_in()`.
        """
        return self.rows_in() * self.cols_in() * self.channels_in()

    def workload_out(self) -> int:
        """
        Returns
        -------
        int
            workload out of layer from port `index` for a
            single featuremap. This is calculated by
            `rows_out()*cols_out()*channels_out()`.
        """
        return self.rows_out() * self.cols_out() * self.channels_out()

    def size_in(self) -> int:
        """
        Returns
        -------
        int
            workload in per stream.
        """
        return self.rows_in() * self.cols_in() * int( self.channels_in() / self.streams_in() )

    def size_out(self) -> int:
        """
        Returns
        -------
        int
            workload out per stream.
        """
        return self.rows_out() * self.cols_out() * int( self.channels_out() / self.streams_out() )

    def width_in(self):
        """
        Returns
        -------
        int
            data width in
        """
        return self.data_width

    def width_out(self):
        """
        Returns
        -------
        int
            data width out
        """
        return self.data_width

    def latency_in(self):
        return abs(self.workload_in()/(self.rate_in()*self.streams_in()))

    def latency_out(self):
        return abs(self.workload_out()/(self.rate_out()*self.streams_out()))

    def latency(self):
        return max(self.latency_in(), self.latency_out())

    def pipeline_depth(self):
        return sum([ self.modules[module].pipeline_depth() for module in self.modules ])

    def wait_depth(self):
        return sum([ self.modules[module].wait_depth() for module in self.modules ])

    def resource(self):
        return {
            "LUT"   : 0,
            "FF"    : 0,
            "BRAM"  : bram_stream_resource_model(self.buffer_depth,self.data_width)*self.streams_in(),
            "DSP"   : 0
        }

    def get_coarse_in_feasible(self, wr_factor=1):
        return get_factors(int(self.channels_in()/wr_factor))

    def get_coarse_out_feasible(self, wr_factor=1):
        return get_factors(int(self.channels_out()/wr_factor))

    def update(self):
        pass

    def layer_info(self, parameters, batch_size=1):
        parameters.batch_size   = batch_size
        parameters.data_width   = self.data_width
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.coarse_in    = self.streams_in()
        parameters.coarse_out   = self.streams_out()

    def get_operations(self):
        return 0

    def layer_info_dict(self):
        # get parameters
        parameter = fpgaconvnet_pb2.parameter()
        self.layer_info(parameter)
        # convert to dictionary
        return MessageToDict(parameter, preserving_proto_field_name=True)

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"edge",str(i)]), label=self.__class__.__name__ ))

        return cluster, "_".join([name,"edge"]), "_".join([name,"edge"])

    def functional_model(self,data,batch_size=1):
        return
