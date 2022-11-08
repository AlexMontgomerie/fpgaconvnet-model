import math
import numpy as np

def int2bits(n):
  return math.ceil(math.log(n, 2))

def utilisation_model(param: dict):
    return {
        "LUT"   : np.array([param["filters"], param["channels"],
            param["data_width"], int2bits(param["channels"])]),
        "FF"    : np.array([param["data_width"], int2bits(param["channels"]),
            int2bits(param["filters"])]),
        "DSP"   : np.array([1]),
        "BRAM"  : np.array([1]),
    }


