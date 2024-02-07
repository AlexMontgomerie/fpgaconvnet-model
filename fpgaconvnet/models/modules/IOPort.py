from dataclasses import dataclass, field

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE
import pydot
import numpy as np

@dataclass
class IOPort(Module):
    num_ports: int
    mem_bw: float
    direction: str
    dma_stream_width: int
    dma_burst_size: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"

    @property
    def port_width(self) -> int:
        return self.data_width*self.num_ports

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["num_ports"] = self.num_ports
        info["mem_bw"] = self.mem_bw
        info["dma_stream_width"] = self.dma_stream_width
        info["dma_burst_size"] = self.dma_burst_size
        # return the info
        return info

    def utilisation_model(self):
        # TODO we should probably use self.dma_stream_width and self.dma_burst_size to calculate the DMA's resource usage
        return {
            "LUT"  : np.array([0]),
            "FF"   : np.array([0]),
            "DSP"  : np.array([0]),
            "BRAM" : np.array([0]),
        }

    def rate_in(self):
        # TODO: should this be 1 on else or infinite?
        return self.mem_bw if self.direction == "in" else 1

    def rate_out(self):
        # TODO: should this be 1 on else or infinite?
        return self.mem_bw if self.direction == "out" else 1

    def get_pred_array(self):
        pass

    def visualise(self, name):
        return pydot.Node(name, label="io_port", shape="polygon",
                sides=4, style="filled", fillcolor="olive", fontsize=MODULE_FONTSIZE)

    def functional_model(self, data):
        return data

    def rsc(self, coef=None, model=None):
        # get the linear model estimation
        # rsc = Module.rsc(self, coef, model)

        if self.regression_model == "linear_regression":
            pass # TODO
        elif self.regression_model == "xgboost" or self.regression_model == "xgboost-kernel":
            pass # TODO

        rsc = {}
        # got theses values from averaging the resource usage of the DMA's across various Vivado projects on ZCU106 device. It definitely needs to be refined.
        rsc['LUT'] = 1160
        rsc['FF'] = 1805
        rsc['BRAM'] = 17
        rsc['DSP']= 0

        # return the resource usage
        return rsc
