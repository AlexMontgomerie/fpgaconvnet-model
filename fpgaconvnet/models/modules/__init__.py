"""
These are the basic building blocks of the accelerator.
"""

MODULE_FONTSIZE=25

from .Module import Module
from .Accum import Accum
from .BatchNorm import BatchNorm
from .Conv import Conv
from .Fork import Fork
from .Glue import Glue
from .Pool import Pool
from .ReLU import ReLU
from .SlidingWindow import SlidingWindow
from .Squeeze import Squeeze
from .Bias import Bias
