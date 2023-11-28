"""
Layers are comprised of modules. They have the same functionality of the equivalent layers of the CNN model.
"""

from enum import Enum, IntEnum, auto

class LayerType(Enum):
    Concat = auto()
    Convolution = auto()
    GlobalPooling = auto()
    HardSwish = auto()
    Pooling = auto()
    ReLU = auto()
    Squeeze = auto()
    ThresholdReLU = auto()

from .layer import LayerBaseMeta, LayerBase

from .concat import ConcatLayerChisel2D
from .convolution import ConvolutionLayerChisel2D, ConvolutionLayerChisel3D, ConvolutionLayerHLS2D, ConvolutionLayerHLS3D
from .global_pooling import GlobalPoolingLayerChisel2D, GlobalPoolingLayerChisel3D
from .hardswish import HardswishLayerChisel2D, HardswishLayerChisel3D, HardswishLayerHLS2D, HardswishLayerHLS3D
from .pooling import PoolingLayerChisel2D, PoolingLayerChisel3D, PoolingLayerHLS2D, PoolingLayerHLS3D
from .relu import ReLULayerChisel2D, ReLULayerChisel3D, ReLULayerHLS2D, ReLULayerHLS3D
from .squeeze import SqueezeLayerChisel2D, SqueezeLayerChisel3D, SqueezeLayerHLS2D, SqueezeLayerHLS3D
from .threshold_relu import ThresholdReLULayerChisel2D, ThresholdReLULayerChisel3D, ThresholdReLULayerHLS2D, ThresholdReLULayerHLS3D
from .split import SplitLayerChisel2D
