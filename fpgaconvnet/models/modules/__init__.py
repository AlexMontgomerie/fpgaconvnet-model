"""
These are the basic building blocks of the accelerator.
"""

MODULE_FONTSIZE=25
MODULE_3D_FONTSIZE=25

from .Module import Module
from .Module3D import Module3D
from .Accum import Accum
from .Accum import Accum3D
from .BatchNorm import BatchNorm
from .Conv import Conv
from .Conv import Conv3D
from .Fork import Fork
from .Fork import Fork3D
from .Glue import Glue
from .Glue import Glue3D
from .Pool import Pool
from .Pool import Pool3D
from .ReLU import ReLU
from .ReLU3D import ReLU3D
from .SlidingWindow import SlidingWindow
from .SlidingWindow import SlidingWindow3D
from .Squeeze import Squeeze
from .Bias import Bias
from .Bias import Bias3D
from .VectorDot import VectorDot
from .VectorDot import VectorDot3D
from .AveragePool import AveragePool
