import numpy as np
import math
import os
from dataclasses import dataclass, field

from fpgaconvnet.models.modules import Module
from fpgaconvnet.tools.resource_model import dsp_multiplier_resource_model

@dataclass
class VectorDot(Module):
    filters: int
    fine: int

    def __post_init__(self):
        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/vector_dot_lutlogic.npy"))

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

    def rate_in(self):
        return 1.0/float(self.filters)


    def rsc(self, coef=None):

        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef

        # LUT
        lut_model = np.array([self.fine, self.filters])
        # FF
        ff = self.int2bits(self.filters)
        # DSP
        dsp = dsp_multiplier_resource_model(self.data_width, self.data_width)

        # return utilisation
        return {
          "LUT"  : int(np.dot(lut_model, self.rsc_coef["LUT"])),
          "BRAM" : 0,
          "DSP"  : dsp,
          "FF"   : ff,
        }

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


