import numpy as np

from fpgaconvnet.models.modules.chisel import int2bits

def utilisation_model(param: dict):
    return {
        "Logic_LUT" : np.array([
            pow(param["kernel_size"][0]*param["kernel_size"][1], 2),
            param["kernel_size"][0]*param["kernel_size"][1],
            param["kernel_size"][0]*param["kernel_size"][1]*param["coarse"],
        ]),
        "LUT_RAM"   : np.array([0]),
        "LUT_SR"    : np.array([0]),
        "FF"    : np.array([
            param["data_width"],
            param["data_width"]*param["kernel_size"][0]*param["kernel_size"][1],
            param["data_width"]*param["kernel_size"][0]*param["kernel_size"][1]*param["coarse"],
        ]),
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
            "LUT"   : rsc["Logic_LUT"],
            "FF"    : rsc["FF"],
            "DSP"   : 0,
            "BRAM"  : 0
        }


