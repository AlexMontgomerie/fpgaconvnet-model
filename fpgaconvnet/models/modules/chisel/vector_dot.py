import numpy as np

from fpgaconvnet.models.modules.chisel import int2bits
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model

def utilisation_model(param: dict):
    return {
        "Logic_LUT" : np.array([
            param["fine"], param["data_width"],
        ]),
        "LUT_RAM"   : np.array([0]),
        "LUT_SR"    : np.array([0]),
        "FF"    : np.array([
            int2bits(param["filters"]),
        ]),
        "DSP"       : np.array([param["fine"]]),
        "BRAM36"    : np.array([0]),
        "BRAM18"    : np.array([0]),
    }

def rsc(param: dict, coef: dict):

        # get the utilisation model
        util_model =  utilisation_model(param)

        # get the linear model for this module
        rsc = { rsc_type: int(np.dot(util_model[rsc_type],
            coef[rsc_type])) for rsc_type in coef.keys()}

        # get the dsp usage
        dsp = param["fine"]*dsp_multiplier_resource_model(
                param["data_width"], param["data_width"])

        # return the resource usage
        return {
            "LUT"   : rsc["Logic_LUT"],
            "FF"    : rsc["FF"],
            "DSP"   : dsp,
            "BRAM"  : 0
        }


