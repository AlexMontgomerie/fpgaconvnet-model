"""
Layers are comprised of modules. They have the same functionality of the equivalent layers of the CNN model.
"""

from dataclasses import dataclass

from .layer import LayerBaseMeta, LayerBase
from .MultiPortLayer import MultiPortLayerBase, MultiPortLayer, MultiPortLayer3D

class Layer:
    pass

class Layer3D:
    pass


from .convolution import ConvolutionLayerChisel
from .global_pooling import GlobalPoolingLayerChisel2D, GlobalPoolingLayerChisel3D
from .hardswish import HardswishLayerChisel2D, HardswishLayerChisel3D, HardswishLayerHLS2D, HardswishLayerHLS3D
from .pooling import PoolingLayerChisel2D, PoolingLayerChisel3D, PoolingLayerHLS2D, PoolingLayerHLS3D
from .relu import ReLULayerChisel2D, ReLULayerChisel3D, ReLULayerHLS2D, ReLULayerHLS3D
from .squeeze import SqueezeLayerChisel2D, SqueezeLayerChisel3D, SqueezeLayerHLS2D, SqueezeLayerHLS3D
from .threshold_relu import ThresholdReLULayerChisel2D, ThresholdReLULayerChisel3D, ThresholdReLULayerHLS2D, ThresholdReLULayerHLS3D

from .BatchNormLayer import BatchNormLayer
from .InnerProductLayer import InnerProductLayer
# from .InnerProductSparseLayer import InnerProductSparseLayer
from .PoolingLayer import PoolingLayer
from .HardswishLayer import HardswishLayer
from .ReLULayer import ReLULayer
from .ThresholdedReLULayer import ThresholdedReLULayer
from .SqueezeLayer import SqueezeLayer
from .SplitLayer import SplitLayer
from .ConcatLayer import ConcatLayer
from .EltWiseLayer import EltWiseLayer
from .GlobalPoolingLayer import GlobalPoolingLayer
from .ReSizeLayer import ReSizeLayer
from .HardswishLayer import HardswishLayer
from .ChopLayer import ChopLayer

# 3D layers
from .InnerProductLayer3D import InnerProductLayer3D
from .PoolingLayer3D import PoolingLayer3D
from .ActivationLayer3D import ActivationLayer3D
from .ReLULayer3D import ReLULayer3D
from .SqueezeLayer3D import SqueezeLayer3D
from .EltWiseLayer3D import EltWiseLayer3D
from .GlobalPoolingLayer3D import GlobalPoolingLayer3D
from .HardswishLayer3D import HardswishLayer3D

@dataclass
class LayerFlag:
    dimensionality: int = 2
    sparsity: bool = False
    data_packing: bool = True
    latency: bool = False
    uram: bool = False
    backend: str = "chisel"
