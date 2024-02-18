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

CHISEL_RSC_TYPES: list[str] = [ "LUT", "FF", "BRAM" ]

from .module import ModuleBaseMeta, ModuleBase, Port, ModuleHLSBase, ModuleHLS3DBase, ModuleChiselBase
# from .resources import ResourceModel, eval_resource_model, get_cached_resource_model

from .accum import AccumChisel, AccumHLS, AccumHLS3D
from .bias import BiasChisel, BiasHLS, BiasHLS3D
from .concat import ConcatChisel
from .conv import ConvHLS, ConvHLS3D
from .eltwise import EltwiseChisel
from .fork import ForkChisel, ForkHLS, ForkHLS3D
from .global_pool import GlobalPoolChisel
from .glue import GlueChisel, GlueHLS, GlueHLS3D
from .hardswish import HardswishChisel
from .pad import PadChisel
from .pool import PoolChisel, PoolHLS, PoolHLS3D
from .relu import ReLUChisel, ReLUHLS, ReLUHLS3D
from .resize import ResizeChisel
from .shift_scale import ShiftScaleChisel
from .sliding_window import SlidingWindowChisel, SlidingWindowHLS, SlidingWindowHLS3D
from .sparse_vector_dot import SparseVectorDotChisel
from .sparse_vector_multiply import SparseVectorMultiplyChisel
from .squeeze import SqueezeChisel, SqueezeHLS, SqueezeHLS3D
from .stride import StrideChisel
from .threshold_relu import ThresholdedReLUChisel
from .vector_dot import VectorDotChisel

