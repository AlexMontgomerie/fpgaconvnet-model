from typing import List
import math
import pydot
import collections
from google.protobuf.json_format import MessageToDict
import numpy as np
from dataclasses import dataclass, field

from fpgaconvnet.models.layers.utils import get_factors
from fpgaconvnet.models.layers.utils import balance_module_rates

from fpgaconvnet.tools.resource_analytical_model import bram_stream_resource_model

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.data_types import FixedPoint

@dataclass
class MultiPortLayer3D:
    """
    Base class for all layer3d models.

    Attributes
    ----------
    buffer_depth: int, default: 0
        depth of incoming fifo buffers for each stream in.
    rows: list int
        row dimension of input featuremap
    cols: list int
        column dimension of input featuremap
    depth: int
        depth dimension of input featuremap
    channels: list int
        channel dimension of input featuremap
    ports_in: int
        number of ports into the layer3d
    ports_out: int
        number of ports out of the layer3d
    coarse_in: list int
        number of parallel streams per port into the layer3d.
    coarse_out: list int
        number of parallel streams per port out of the layer3d.
    data_width: int
        bitwidth of featuremap pixels
    modules: dict
        dictionary of `module` instances that make
        up the layer3d. These modules are used for the
        resource and performance models of the layer3d.
    """

    _rows: List[int]
    _cols: List[int]
    _depth: List[int]
    _channels: List[int]
    _coarse_in: List[int]
    _coarse_out: List[int]
    ports_in: int = field(default=1, init=True)
    ports_out: int = field(default=1, init=True)
    data_t: FixedPoint = field(default=FixedPoint(16,8), init=True)
    modules: dict = field(default_factory=collections.OrderedDict, init=False)

    def __post_init__(self):
        self.buffer_depth = [2]*self.ports_in

    """
    properties
    """

    @property
    def rows(self) -> List[int]:
        return self._rows

    @property
    def cols(self) -> List[int]:
        return self._cols

    @property
    def depth(self) -> List[int]:
        return self._depth

    @property
    def channels(self) -> List[int]:
        return self._channels

    @property
    def coarse_in(self) -> List[int]:
        return self._coarse_in

    @property
    def coarse_out(self) -> List[int]:
        return self._coarse_out

    """
    property setters
    """

    @rows.setter
    def rows(self, val: List[int]) -> None:
        assert(len(val) == self.ports_in)
        self._rows = val
        self.update()

    @cols.setter
    def cols(self, val: List[int]) -> None:
        assert(len(val) == self.ports_in)
        self._cols = val
        self.update()

    @depth.setter
    def depth(self, val: List[int]) -> None:
        assert(len(val) == self.ports_in)
        self._depth = val
        self.update()

    @channels.setter
    def channels(self, val: List[int]) -> None:
        assert(len(val) == self.ports_in)
        self._channels = val
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: List[int]) -> None:
        assert(len(val) == self.ports_in)
        # for i in range(val):
        #     assert(val[i] in self.coarse_in_feasible(port_index=i))
        self._coarse_in = val
        self.coarse_out = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: List[int]) -> None:
        assert(len(val) == self.ports_out)
        # for i in range(val):
        #     assert(val[i] in self.coarse_out_feasible(port_index=i))
        self._coarse_out = val
        self._coarse_in = val
        self.update()

    def rows_in(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port into the layer3d

        Returns
        -------
        int
            row dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.rows[port_index]

    def cols_in(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port into the layer3d

        Returns
        -------
        int
            column dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.cols[port_index]

    def depth_in(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port into the layer3d

        Returns
        -------
        int
            depth dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.depth[port_index]

    def channels_in(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port into the layer3d

        Returns
        -------
        int
            channel dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.channels[port_index]

    def rows_out(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port out of the layer3d

        Returns
        -------
        int
            row dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return self.rows[port_index]

    def cols_out(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port out of the layer3d

        Returns
        -------
        int
            column dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return self.cols[port_index]

    def depth_out(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port out of the layer3d

        Returns
        -------
        int
            depth dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return self.depth[port_index]

    def channels_out(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port out of the layer3d

        Returns
        -------
        int
            channel dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return self.channels[port_index]

    def rates_graph(self):

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

    def rate_in(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port into layer3d

        Returns
        -------
        float
            rate of words into layer3d. As a fraction of a
            clock cycle.

            default is 1.0
        """
        assert(port_index < self.ports_in)
        return abs(balance_module_rates(self.rates_graph())[0,0])

    def rate_out(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port into layer3d

        Returns
        -------
        float
            rate of words out of the layer3d. As a fraction
            of a clock cycle.

            default is 1.0
        """
        assert(port_index < self.ports_out)
        return abs(balance_module_rates(
            self.rates_graph())[len(self.modules.keys())-1,len(self.modules.keys())])

    def streams_in(self, port_index=0):
        """
        Returns
        -------
        int
            number of parallel streams into the layer3d.
        """
        assert(port_index < self.ports_in)
        return self.coarse_in[port_index]

    def streams_out(self, port_index=0):
        """
        Returns
        -------
        int
            number of parallel streams out of the layer3d.
        """
        assert(port_index < self.ports_out)
        return self.coarse_out[port_index]

    def workload_in(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port into layer3d

        Returns
        -------
        int
            workload into layer3d from port `index` for a single
            featuremap. This is calculated by
            `rows_in()*cols_in()*depth_in()*channels_in()`.
        """
        assert(port_index < self.ports_in)
        return self.rows_in(port_index) * self.cols_in(port_index) * self.depth_in(port_index) * self.channels_in(port_index)

    def workload_out(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port out of layer3d

        Returns
        -------
        int
            workload out of layer3d from port `index` for a
            single featuremap. This is calculated by
            `rows_out()*cols_out()*depth_out()*channels_out()`.
        """
        assert(port_index < self.ports_out)
        return self.rows_out(port_index) * self.cols_out(port_index) * self.depth_out(port_index) * self.channels_out(port_index)

    def size_in(self, port_index=0):
        """
        Returns
        -------
        int
            workload in per stream.
        """
        assert(port_index < self.ports_in)
        return self.rows_in(port_index) * self.cols_in(port_index) * self.depth_in(port_index) * int( self.channels_in(port_index) / self.streams_in(port_index) )

    def size_out(self, port_index=0):
        """
        Returns
        -------
        int
            workload out per stream.
        """
        assert(port_index < self.ports_out)
        return self.rows_out(port_index) * self.cols_out(port_index) * self.depth_out(port_index) * int( self.channels_out(port_index) / self.streams_out(port_index) )

    def shape_in(self, port_index=0) -> List[int]: # TODO: add documentation
        return [ self.rows_in(port_index), self.cols_in(port_index), self.depth_in(port_index), self.channels_in(port_index) ]

    def shape_out(self, port_index=0) -> List[int]: # TODO: add documentation
        return [ self.rows_out(port_index), self.cols_out(port_index), self.depth_out(port_index), self.channels_out(port_index) ]

    def width_in(self):
        """
        Returns
        -------
        int
            data width in
        """
        return self.data_t.width

    def width_out(self):
        """
        Returns
        -------
        int
            data width out
        """
        return self.data_t.width

    def latency_in(self):
        return max([
            abs(self.workload_in(i)/(self.rate_in(i)*self.streams_in(i) )) for
            i in range(self.ports_in) ])

    def latency_out(self):
        return max([
            abs(self.workload_out(i)/(self.rate_out(i)*self.streams_out(i)))
            for i in range(self.ports_out) ])

    def latency(self):
        return max(self.latency_in(), self.latency_out())

    def pipeline_depth(self):
        return sum([ self.modules[module].pipeline_depth() for module in self.modules ])

    def wait_depth(self):
        return sum([ self.modules[module].wait_depth() for module in self.modules ])

    def resource(self):
        # bram for fifos
        fifo_bram = sum([bram_stream_resource_model(
                self.buffer_depth[i], self.data_t.width
            )*self.streams_in(i) for i in range(self.ports_in) ])
        return {
            "LUT"   : 0,
            "FF"    : 0,
            "BRAM"  : fifo_bram,
            "DSP"   : 0
        }

    def get_coarse_in_feasible(self, port_index=0):
        assert(port_index < self.ports_in)
        return get_factors(self.channels_in(port_index))

    def get_coarse_out_feasible(self, port_index=0):
        assert(port_index < self.ports_out)
        return get_factors(self.channels_out(port_index))

    def update(self):
        pass

    def layer_info(self, parameters, batch_size=1):
        parameters.batch_size   = batch_size
        parameters.rows_in_array.extend(map(self.rows_in, range(self.ports_in)))
        parameters.cols_in_array.extend(map(self.cols_in, range(self.ports_in)))
        parameters.depth_in_array.extend(map(self.depth_in, range(self.ports_in)))
        parameters.channels_in_array.extend(map(self.channels_in, range(self.ports_in)))
        parameters.rows_out_array.extend(map(self.rows_out, range(self.ports_out)))
        parameters.cols_out_array.extend(map(self.cols_out, range(self.ports_out)))
        parameters.depth_out_array.extend(map(self.depth_out, range(self.ports_out)))
        parameters.channels_out_array.extend(map(self.channels_out, range(self.ports_out)))
        parameters.coarse_in    = self.streams_in()
        parameters.coarse_out   = self.streams_out()
        parameters.ports_in     = self.ports_in
        parameters.ports_out    = self.ports_out
        self.data_t.to_protobuf(parameters.data_t)

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

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"edge",str(i)]), label=self.__class__.__name__ ))

        return cluster, "_".join([name,"edge"]), "_".join([name,"edge"])

    def functional_model(self,data,batch_size=1):
        return
