'''
Base class for all hardware module models.
'''

import numpy as np
import math
import os
import copy
from typing import List
from dataclasses import dataclass, field

RSC_TYPES = ["FF","LUT","DSP","BRAM"]

@dataclass
class Module:
    """
    modules are the fundamental building block for the hardware
    framework. In this base class, performance and resource model
    templates are included, as well as a template for functional
    models. All modules are derived from this base class and contain
    the same methods.

    Attributes
    ----------
    rows: int
        row dimension of input featuremap
    cols: int
        column dimension of input featuremap
    channels: int
        channel dimension of input featuremap
    data_width: int
        bitwidth of featuremap pixels
    rsc_coef: list
        list of resource model coefficients. Corresponds
        to `LUT`, `BRAM`, `DSP` and `FF` resources in
        that order.

    .. note::
        The model expects that the module is run for a single three
        dimensional featuremap. For intermediate modules within a layer,
        they may not be operating on a three dimensional tensor, and
        so the `rows`, `cols` and `channels` attributes are representative
        of the tensor if it was flattened to three dimensions.
    """
    rows: int
    cols: int
    channels: int
    data_width: int = field(default=16, init=False)
    rsc_coef: dict = field(default_factory=lambda: {"FF": [], "LUT": [], "DSP": [], "BRAM": []}, init=False)

    def module_info(self):
        """
        creates a dictionary containing information and
        parameters for the module.
        """
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def utilisation_model(self):
        """
        Returns
        -------
        dict
            utilisation of resources model. Defaults
            to zero resources.
        """
        return {
            "LUT"  : np.array([0]),
            "DSP"  : np.array([0]),
            "BRAM" : np.array([0]),
            "FF"   : np.array([0]),
        }

    def rows_in(self):
        """
        Returns
        -------
        int
            row dimension of the input featuremap
        """
        return self.rows

    def cols_in(self):
        """
        Returns
        -------
        int
            column dimension of the input featuremap
        """
        return self.cols

    def channels_in(self):
        """
        Returns
        -------
        int
            channel dimension of the input featuremap
        """
        return self.channels

    def rows_out(self):
        """
        Returns
        -------
        int
            row dimension of the output featuremap
        """
        return self.rows

    def cols_out(self):
        """
        Returns
        -------
        int
            column dimension of the output featuremap
        """
        return self.cols

    def channels_out(self):
        """
        Returns
        -------
        int
            channel dimension of the output featuremap
        """
        return self.channels

    def rate_in(self):
        """
        Returns
        -------
        float
            rate of words into module. As a fraction of a
            clock cycle.

            default is 1.0
        """
        return 1.0

    def rate_out(self):
        """
        Returns
        -------
        float
            rate of words out of the module. As a fraction
            of a clock cycle.

            default is 1.0
        """
        return 1.0

    def latency(self):
        """
        Returns
        -------
        int
            calculates the number of clock cycles latency
            it takes for the module to process a featuremap.
            First latency in and latency out is calculated,
            then the latency of the module is the largest of
            the two.
        """
        latency_in  = int((self.rows_in() *self.cols_in() *self.channels_in() )/self.rate_in() )
        latency_out = int((self.rows_out()*self.cols_out()*self.channels_out())/self.rate_out())
        return max(latency_in,latency_out)

    def pipeline_depth(self):
        """
        Returns
        -------
        int
           depth of the pipeline for the module in clock
           cycles.

           default is 0.
        """
        return 0

    def int2bits(self, n):
        """
        helper function to get number of bits for integer
        """
        return math.ceil(math.log(n, 2))

    def rsc(self, coef=None):
        """
        Returns
        -------
        dict
            estimated resource usage of the module. Uses the
            resource coefficients for the estimate.
        """

        # get the utilisation model
        utilisation_model = self.utilisation_model()

        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef

        # return the linear model estimation
        return {
          "LUT"  : int(np.dot(utilisation_model["LUT"], coef["LUT"])),
          "BRAM" : int(np.dot(utilisation_model["BRAM"], coef["BRAM"])),
          "DSP"  : int(np.dot(utilisation_model["DSP"], coef["DSP"])),
          "FF"   : int(np.dot(utilisation_model["FF"], coef["FF"])),
        }

    def functional_model(self,data):
        """
        functional model of the module. Used for verification
        of hardware modules.

        Returns
        -------
        np.array
        """
        return data
