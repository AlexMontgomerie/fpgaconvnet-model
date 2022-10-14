"""
These are the basic building blocks of the accelerator.
"""

MODULE_FONTSIZE=25
MODULE_3D_FONTSIZE=25

from .Module import Module
from .Module3D import Module3D
from .Accum import Accum
from .BatchNorm import BatchNorm
from .Conv import Conv
from .Fork import Fork
from .Glue import Glue
from .Pool import Pool
from .ReLU import ReLU
from .ReLU3D import ReLU3D
from .SlidingWindow import SlidingWindow
from .Squeeze import Squeeze
from .Bias import Bias
from .VectorDot import VectorDot
from .AveragePool import AveragePool
from .Concat import Concat
from .EltWise import EltWise
from .Stride import Stride
