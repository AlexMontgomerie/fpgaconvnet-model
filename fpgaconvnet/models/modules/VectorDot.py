import numpy as np
import math
import os
from dataclasses import dataclass, field

from fpgaconvnet.models.modules import int2bits, Module
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model, queue_lutram_resource_model

@dataclass
class VectorDot(Module):
    filters: int
    fine: int
    backend: str = "chisel"
    regression_model: str = "linear_regression"
    weight_width: int = field(default=16, init=False)
    acc_width: int = field(default=32, init=False)
    streams: int = 1

    def pipeline_depth(self):
        return int(math.log(self.fine, 2)) + 1

    def channels_out(self):
        return self.filters

    def rate_in(self):
        return 1.0/float(self.filters)

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'fine'    : self.fine,
            'filters'    : self.filters,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def utilisation_model(self):
        if self.backend == "hls":
            pass
        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.fine, self.data_width, self.weight_width,
                    self.streams*self.data_width*self.fine,
                    self.streams*self.weight_width*self.fine,
                    self.streams*self.acc_width*self.fine, # adder tree
                    self.filters, # ready logic
                    int2bits(self.filters), # filter counter
                    1,
                ]),
                "LUT_RAM"   : np.array([
                    queue_lutram_resource_model(
                        int2bits(self.fine)+1, self.streams*self.acc_width), # buffer
                    1,
                ]),
                "LUT_SR"    : np.array([
                    int2bits(self.fine)+1, # tree buffer valid
                ]),
                "FF"    : np.array([
                    self.acc_width, # output buffer TODO
                    self.filters, self.fine, # parameters
                    int2bits(self.filters), # filter counter
                    int2bits(self.fine)+1, # tree buffer valid
                    self.streams*self.acc_width*self.fine, # adder tree reg
                    self.streams*self.acc_width, # output buffer
                    1,
                ]),
                "DSP"       : np.array([self.streams*self.fine]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }
        else:
            raise ValueError(f"{self.backend} backend not supported")

    def get_pred_array(self):
        return np.array([
            self.data_width, self.data_width//2,
            self.filters, self.fine, self.streams,
            self.acc_width, self.acc_width//2,
            self.weight_width, self.weight_width//2,
        ]).reshape(1,-1)

    def rsc(self,coef=None, model=None):

        # get the linear model estimation
        rsc = Module.rsc(self, coef, model)

        if self.regression_model == "linear_regression":
            # get the dsp usage
            dsp = self.streams*self.fine*dsp_multiplier_resource_model(
                    self.data_width, self.weight_width)
            rsc["DSP"] = dsp

        return rsc

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data, weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.fine    , "ERROR: invalid column dimension"
        # check input dimensionality
        assert weights.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert weights.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert weights.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[3] == self.filters , "ERROR: invalid channel dimension"
        assert weights.shape[4] == self.fine    , "ERROR: invalid column dimension"


        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels,
            self.filters),dtype=object)

        for index,_ in np.ndenumerate(weights):
            if index[4] == 0:
                out[
                    index[0],
                    index[1],
                    index[2],
                    index[3]
                ] = weights[index] * data[
                        index[0],
                        index[1],
                        index[2],
                        index[4]
                    ]
            else:
                out[
                    index[0],
                    index[1],
                    index[2],
                    index[3]
                ] += weights[index] * data[
                        index[0],
                        index[1],
                        index[2],
                        index[4]
                    ]

        return out


