"""
These are the basic building blocks of the accelerator.
"""

MODULE_FONTSIZE=25
MODULE_3D_FONTSIZE=25

import math

def int2bits(n):
    """
    helper function to get number of bits for integer
    """
    return math.ceil(math.log(n, 2))

from .Module import Module
from .Accum import Accum
from .ShiftScale import ShiftScale
from .Conv import Conv
from .Fork import Fork
from .Glue import Glue
from .Pool import Pool
from .Pad import Pad
from .MaxPool import MaxPool
# from .Pool import AvgPool
from .ReLU import ReLU
from .Hardswish import Hardswish
from .ReSize import ReSize
from .ThresholdedReLU import ThresholdedReLU
from .SlidingWindow import SlidingWindow
from .Squeeze import Squeeze
from .Bias import Bias
from .VectorDot import VectorDot
from .SparseVectorDot import SparseVectorDot
from .GlobalPool import GlobalPool
from .Concat import Concat
from .EltWise import EltWise
from .Stride import Stride

# 3D modules
from .Module3D import Module3D
from .Accum3D import Accum3D
from .Conv3D import Conv3D
from .Fork3D import Fork3D
from .Glue3D import Glue3D
from .Pool3D import Pool3D
from .Activation3D import Activation3D
from .ReLU3D import ReLU3D
from .Hardswish3D import Hardswish3D
from .ReSize3D import ReSize3D
from .SlidingWindow3D import SlidingWindow3D
from .Squeeze3D import Squeeze3D
from .Bias3D import Bias3D
from .VectorDot3D import VectorDot3D
from .GlobalPool3D import GlobalPool3D
from .EltWise3D import EltWise3D
from .Concat3D import Concat3D
from .Pad3D import Pad3D
from .ShiftScale3D import ShiftScale3D
