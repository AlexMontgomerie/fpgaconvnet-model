import numpy as np
import math
import os
from dataclasses import dataclass, field

from fpgaconvnet.models.modules import int2bits, Module3D
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model

@dataclass
class VectorDot3D(Module3D):
    filters: int
    fine: int
    weight_width: int = field(default=16, init=False)
    acc_width: int = field(default=32, init=False)
    backend: str = "chisel"

    def __post_init__(self):

        # get the cache path
        rsc_cache_path = os.path.dirname(__file__) + \
                f"/../../coefficients/{self.backend}"

        # iterate over resource types
        self.rsc_coef = {}
        for rsc_type in self.utilisation_model():
            # load the resource coefficients from the 2D version
            coef_path = os.path.join(rsc_cache_path, f"{self.__class__.__name__.split('3D')[0]}_{rsc_type}.npy".lower())
            self.rsc_coef[rsc_type] = np.load(coef_path)

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'depth'      : self.depth_in(),
            'channels'  : self.channels_in(),
            'fine'    : self.fine,
            'filters'    : self.filters,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'depth_out'      : self.depth_out(),
            'channels_out'  : self.channels_out()
        }

    def rate_in(self):
        return 1.0/float(self.filters)

    def memory_usage(self):
        if self.backend == "chisel":
            return self.data_width*(int2bits(self.fine)+3)
        else:
            raise NotImplementedError

    def utilisation_model(self):
        if self.backend == "hls":
            pass
        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.fine, self.data_width, self.weight_width,
                    self.data_width*self.fine,
                    self.weight_width*self.fine,
                    self.acc_width*self.fine, # adder tree
                    self.filters, # ready logic
                    int2bits(self.filters), # filter counter
                    1,
                ]),
                "LUT_RAM"   : np.array([
                    self.acc_width*(int2bits(self.fine)+3), # tree buffer
                    # self.acc_width*self.fine, # tree buffer
                    self.acc_width, # output buffer
                ]),
                "LUT_SR"    : np.array([
                    int2bits(self.fine)+1, # tree buffer valid
                ]),
                "FF"    : np.array([
                    self.acc_width, # output buffer TODO
                    int2bits(self.filters), # filter counter
                    int2bits(self.fine)+1, # tree buffer valid
                    self.acc_width*self.fine, # adder tree reg
                    self.data_width*self.fine, # adder tree reg
                    self.weight_width*self.fine, # adder tree reg
                    1,
                ]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }
        else:
            raise ValueError(f"{self.backend} backend not supported")

    def rsc(self,coef=None):

        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef

        # get the linear model estimation
        rsc = Module3D.rsc(self, coef)

        # get the dsp usage
        dsp = self.fine*dsp_multiplier_resource_model(
                self.data_width, self.data_width)
        rsc["DSP"] = dsp

        return rsc

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data, weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.depth   , "ERROR: invalid depth dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[4] == self.fine    , "ERROR: invalid column dimension"
        # check input dimensionality
        assert weights.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert weights.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert weights.shape[2] == self.depth   , "ERROR: invalid depth dimension"
        assert weights.shape[3] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[4] == self.filters , "ERROR: invalid channel dimension"
        assert weights.shape[5] == self.fine    , "ERROR: invalid column dimension"


        out = np.ndarray((
            self.rows,
            self.cols,
            self.depth,
            self.channels,
            self.filters),dtype=object)

        for index,_ in np.ndenumerate(weights):
            if index[5] == 0:
                out[
                    index[0],
                    index[1],
                    index[2],
                    index[3],
                    index[4]
                ] = weights[index] * data[
                        index[0],
                        index[1],
                        index[2],
                        index[3],
                        index[5]
                    ]
            else:
                out[
                    index[0],
                    index[1],
                    index[2],
                    index[3],
                    index[4]
                ] += weights[index] * data[
                        index[0],
                        index[1],
                        index[2],
                        index[3],
                        index[5]
                    ]

        return out


