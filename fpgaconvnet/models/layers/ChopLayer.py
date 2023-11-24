"""
The chop/fork/branch layer.
Takes one stream input and outputs several streams using the fork module.
"""

from dataclasses import dataclass, field
from typing import Any, List

import pydot

from fpgaconvnet.models.layers import MultiPortLayer
# from fpgaconvnet.models.modules import Fork


@dataclass(kw_only=True)
class ChopLayer(MultiPortLayer):
    split: List[int]
    coarse: int = 1
    ports_out: int = 1
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    def __post_init__(self):

        # call parent post init
        super().__post_init__()

        # backend flag
        assert (self.backend in ["hls", "chisel"], f"{self.backend} is an invalid backend")

        # regression model
        assert(self.regression_model in ["linear_regression", "xgboost"],
                f"{self.regression_model} is an invalid regression model")

        self.mem_bw_out = [100.0/self.ports_out] * self.ports_out

        # init modules
        #One fork module, fork coarse_out corresponds to number of layer output ports
        # self.modules["fork"] = Fork( self.rows_in(), self.cols_in(),
        #         self.channels_in(), 1, self.ports_out, backend=self.backend, regression_model=self.regression_model)

        # update the modules
        self.update()

    def __setattr__(self, name: str, value: Any) -> None:

        if not hasattr(self, "is_init"):
            super().__setattr__(name, value)
            return

        match name:
            case "coarse":
                assert(value in self.get_coarse_in_feasible())
                super().__setattr__("coarse", value)
                self.update()
            case "coarse_in":
                assert(value in self.get_coarse_in_feasible())
                super().__setattr__("coarse_in", [value])
                self.update()
            case "coarse_out":
                assert(value in self.get_coarse_out_feasible())
                super().__setattr__("coarse_out", [value]*self.ports_out)
                self.update()
            case _:
                super().__setattr__(name, value)

    def get_operations(self):
        raise NotImplementedError

    def streams_in(self, port_index=0):
        """
        Returns
        -------
        int
            number of parallel streams into the layer.
        """
        assert(port_index < self.ports_in)
        return self._coarse

    def rows_out(self, port_index=0) -> int:
        return self.rows[0]

    def cols_out(self, port_index=0) -> int:
        return self.cols[0]

    def channels_out(self, port_index=0) -> int:
        return self.split[port_index]

    def rate_in(self, port_index=0):
        assert(port_index < self.ports_in)
        return 1.0

    def rate_out(self, port_index=0):
        assert(port_index < self.ports_out)
        return self.channels_out(port_index)/self.channels_in()

    def layer_info(self,parameters,batch_size=1):
        MultiPortLayer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.split.extend(self.split)
        # remove the repeated rows, cols and channels
        del parameters.rows_in_array[:]
        del parameters.cols_in_array[:]
        del parameters.channels_in_array[:]
        del parameters.rows_out_array[:]
        del parameters.cols_out_array[:]

    def update(self):
        pass

    def resource(self):

        #Total
        return {
            "LUT"   : 0,
            "FF"    : 0,
            "BRAM"  : 0,
            "DSP"   : 0
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in self.coarse_in:
            cluster.add_node(pydot.Node( "_".join([name,"chop",str(i)]), label="chop" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"chop",str(i)]) for i in range(self.coarse) ]
        nodes_out = [ "_".join([name,"chop",str(i)]) for i in range(self.ports_out) ]

        return cluster, nodes_in, nodes_out


