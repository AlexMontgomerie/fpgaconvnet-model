import numpy as np

from fpgaconvnet.models.modules.chisel import int2bits
from fpgaconvnet.tools.resource_analytical_model import bram_stream_resource_model

def utilisation_model(param: dict):
    return {
        "Logic_LUT" : np.array([1]),
        "LUT_RAM"   : np.array([1]),
        "LUT_SR"    : np.array([1]),
        "FF"        : np.array([1]),
        "DSP"       : np.array([0]),
        "BRAM36"    : np.array([0]),
        "BRAM18"    : np.array([0]),
    }

def rsc(param: dict, coef: dict):

        # get the utilisation model
        util_model =  utilisation_model(param)

        # get the linear model for this module
        rsc = { rsc_type: int(np.dot(util_model[rsc_type],
            coef[rsc_type])) for rsc_type in coef.keys()}

        # return the resource usage
        return {
            "LUT"   : rsc["Logic_LUT"] + rsc["LUT_RAM"] + rsc["LUT_SR"],
            "FF"    : rsc["FF"],
            "DSP"   : 0,
            "BRAM"  : 0
        }

