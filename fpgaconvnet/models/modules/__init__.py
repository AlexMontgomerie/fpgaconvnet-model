"""
These are the basic building blocks of the accelerator.
"""

MODULE_FONTSIZE=25
MODULE_3D_FONTSIZE=25

from .Module import Module
from .Module3D import Module3D
from .Accum import Accum
from .Accum3D import Accum3D
from .BatchNorm import BatchNorm
from .Conv import Conv
from .Conv3D import Conv3D
from .Fork import Fork
from .Fork3D import Fork3D
from .Glue import Glue
from .Glue3D import Glue3D
from .Pool import Pool
from .Pool3D import Pool3D
from .ReLU import ReLU
from .ReLU3D import ReLU3D
from .SlidingWindow import SlidingWindow
from .SlidingWindow3D import SlidingWindow3D
from .Squeeze import Squeeze
from .Bias import Bias
from .Bias3D import Bias3D
from .VectorDot import VectorDot
from .VectorDot3D import VectorDot3D
from .AveragePool import AveragePool
