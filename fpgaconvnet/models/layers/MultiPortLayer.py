import collections
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pydot
from google.protobuf.json_format import MessageToDict

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.data_types import FixedPoint
from fpgaconvnet.models.layers.Layer import LayerBaseMeta
from fpgaconvnet.models.layers.utils import balance_module_rates, get_factors


@dataclass
class MultiPortLayerBase(metaclass=LayerBaseMeta):
    coarse_in: int = field(default_factory=lambda: [1], init=False)
    coarse_out: int = field(default_factory=lambda: [1], init=False)
    mem_bw_in: float = field(default_factory=lambda: [100.0], init=True)
    mem_bw_out: float = field(default_factory=lambda: [100.0], init=True)
    ports_in: int = field(default=1, init=True)
    ports_out: int = field(default=1, init=True)
    data_t: FixedPoint = field(default_factory=lambda: FixedPoint(16,8), init=True)
    modules: dict = field(default_factory=collections.OrderedDict, init=False)

    def __post_init__(self):
        self.input_t = self.data_t
        self.output_t = self.data_t
        self.stream_inputs = [False]*self.ports_in
        self.stream_outputs = [False]*self.ports_out
        self.buffer_depth = [2]*self.ports_in
        self.is_init = True

    def __setattr__(self, name, value):
        """
        Set the value of an attribute and update the layer.

        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to set the attribute to.

        Raises:
            AssertionError: If the value is not feasible for the attribute.

        Returns:
            None
        """

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        match name:
            case "coarse_in":
                assert(len(value) == self.ports_in)
                super().__setattr__(name, value)
                self.update()

            case "coarse_out":
                assert(len(value) == self.ports_out)
                super().__setattr__(name, value)
                self.update()

            case _:
                super().__setattr__(name, value)

    @abstractmethod
    def shape_in(self, port_index=0) -> List[int]:
        pass

    @abstractmethod
    def shape_out(self, port_index=0) -> List[int]:
        pass

    @abstractmethod
    def rate_in(self, port_index=0) -> float:
        pass

    @abstractmethod
    def rate_out(self, port_index=0) -> float:
        pass

    def width_in(self):
        raise NotImplementedError

    def width_out(self):
        raise NotImplementedError

    def streams_in(self, port_index=0):
        """
        Returns
        -------
        int
            number of parallel streams into the layer.
        """
        assert(port_index < self.ports_in)
        return self.coarse_in[port_index]

    def streams_out(self, port_index=0):
        """
        Returns
        -------
        int
            number of parallel streams out of the layer.
        """
        assert(port_index < self.ports_out)
        return self.coarse_out[port_index]

    def workload_in(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port into layer

        Returns
        -------
        int
            The total number of elements in the input tensor of this layer.
        """
        assert(port_index < self.ports_in)
        return np.prod(self.shape_in(port_index))

    def workload_out(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port out of layer

        Returns
        -------
        int
            The total number of elements in the output tensor of this layer.
        """
        assert(port_index < self.ports_out)
        return np.prod(self.shape_out(port_index))

    def size_in(self, port_index=0):
        """
        Returns
        -------
        int
            workload in per stream.
        """
        assert(port_index < self.ports_in)
        return self.workload_in(port_index) / self.streams_in(port_index)

    def size_out(self, port_index=0):
        """
        Returns
        -------
        int
            workload out per stream.
        """
        assert(port_index < self.ports_out)
        return self.workload_out(port_index) / self.streams_out(port_index)

    def latency_in(self):
        return max([
            abs(self.workload_in(i)/(min(self.mem_bw_in[i], self.rate_in(i)*self.streams_in(i)))) for
            i in range(self.ports_in) ])

    def latency_out(self):
        return max([
            abs(self.workload_out(i)/(min(self.mem_bw_out[i], self.rate_out(i)*self.streams_out(i))))
            for i in range(self.ports_out) ])

    def latency(self):
        return max(self.latency_in(), self.latency_out())

    def pipeline_depth(self):
        return sum([ self.modules[module].pipeline_depth() for module in self.modules ])

    def wait_depth(self):
        return sum([ self.modules[module].wait_depth() for module in self.modules ])

    @abstractmethod
    def resource(self):
        pass

    def memory_bandwidth(self):
        return {
            "in"  : sum([min(self.mem_bw_in[i], self.rate_in(i)*self.streams_in(i)) for i in range(self.ports_in)]),
            "out" : sum([min(self.mem_bw_out[i], self.rate_out(i)*self.streams_out(i)) for i in range(self.ports_out)])
        }

    @abstractmethod
    def get_coarse_in_feasible(self):
        pass

    @abstractmethod
    def get_coarse_out_feasible(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def layer_info(self, parameters, batch_size=1):
        parameters.batch_size   = batch_size
        parameters.mem_bw_in_array.extend([self.mem_bw_in[i] for i in range(self.ports_in)])
        parameters.mem_bw_out_array.extend([self.mem_bw_out[i] for i in range(self.ports_out)])
        parameters.coarse_in    = self.coarse_in
        parameters.coarse_out   = self.coarse_out
        parameters.ports_in     = self.ports_in
        parameters.ports_out    = self.ports_out
        # parameters.stream_inputs.extend(self.stream_inputs)
        # parameters.stream_outputs.extend(self.stream_outputs)
        self.data_t.to_protobuf(parameters.data_t)

    def get_operations(self):
        return 0

    def get_sparse_operations(self):
        return self.get_operations()

    def layer_info_dict(self):
        # get parameters
        parameter = fpgaconvnet_pb2.parameter()
        self.layer_info(parameter)
        # convert to dictionary
        return MessageToDict(parameter, preserving_proto_field_name=True)

    def visualise(self, name):
        raise NotImplementedError

    @abstractmethod
    def functional_model(self, data, batch_size=1):
        raise NotImplementedError(f"Functional model not implemented for multiport layer type: {self.__class__.__name__}")

@dataclass(kw_only=True)
class MultiPortLayer(MultiPortLayerBase):
    rows: List[int]
    cols: List[int]
    channels: List[int]

    def rows_in(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port into the layer

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
            index of port into the layer

        Returns
        -------
        int
            column dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.cols[port_index]

    def channels_in(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port into the layer

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
            index of port out of the layer

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
            index of port out of the layer

        Returns
        -------
        int
            column dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return self.cols[port_index]

    def channels_out(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port out of the layer

        Returns
        -------
        int
            channel dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return self.channels[port_index]

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

    def rate_in(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port into layer

        Returns
        -------
        float
            rate of words into layer. As a fraction of a
            clock cycle.

            default is 1.0
        """
        assert(port_index < self.ports_in)
        return abs(balance_module_rates(self.build_rates_graph())[0,0])

    def rate_out(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port into layer

        Returns
        -------
        float
            rate of words out of the layer. As a fraction
            of a clock cycle.

            default is 1.0
        """
        assert(port_index < self.ports_out)
        return abs(balance_module_rates(
            self.build_rates_graph())[len(self.modules.keys())-1,len(self.modules.keys())])

    def shape_in(self, port_index=0) -> List[int]:
        return [ self.rows_in(port_index), self.cols_in(port_index), self.channels_in(port_index) ]

    def shape_out(self, port_index=0) -> List[int]:
        return [ self.rows_out(port_index), self.cols_out(port_index), self.channels_out(port_index) ]

    def width_in(self):
        return self.data_t.width

    def width_out(self):
        return self.data_t.width

    def get_coarse_in_feasible(self, port_index=0):
        assert(port_index < self.ports_in)
        return get_factors(self.channels_in(port_index))

    def get_coarse_out_feasible(self, port_index=0):
        assert(port_index < self.ports_out)
        return get_factors(self.channels_out(port_index))

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(self, parameters, batch_size)
        parameters.rows_in_array.extend(map(self.rows_in, range(self.ports_in)))
        parameters.cols_in_array.extend(map(self.cols_in, range(self.ports_in)))
        parameters.channels_in_array.extend(map(self.channels_in, range(self.ports_in)))
        parameters.rows_out_array.extend(map(self.rows_out, range(self.ports_out)))
        parameters.cols_out_array.extend(map(self.cols_out, range(self.ports_out)))
        parameters.channels_out_array.extend(map(self.channels_out, range(self.ports_out)))

    def visualise(self, name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"edge",str(i)]), label=self.__class__.__name__ ))

        return cluster, "_".join([name,"edge"]), "_".join([name,"edge"])

@dataclass(kw_only=True)
class MultiPortLayer3D(MultiPortLayer):
    depth: List[int]

    def depth_in(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port into the layer

        Returns
        -------
        int
            depth dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.depth[port_index]

    def depth_out(self, port_index=0):
        """
        Parameters
        ----------
        port_index: int
            index of port out of the layer

        Returns
        -------
        int
            depth dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return self.depth[port_index]

    def shape_in(self, port_index=0) -> List[int]:
        return [ self.rows_in(port_index), self.cols_in(port_index), self.depth_in(port_index), self.channels_in(port_index) ]

    def shape_out(self, port_index=0) -> List[int]:
        return [ self.rows_out(port_index), self.cols_out(port_index), self.depth_out(port_index), self.channels_out(port_index) ]

    def layer_info(self, parameters, batch_size=1):
        super().layer_info(self, parameters, batch_size)
        parameters.depth_in_array.extend(map(self.depth_in, range(self.ports_in)))
        parameters.depth_out_array.extend(map(self.depth_out, range(self.ports_out)))