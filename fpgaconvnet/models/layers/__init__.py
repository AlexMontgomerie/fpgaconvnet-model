"""
Layers are comprised of modules. They have the same functionality of the equivalent layers of the CNN model.
"""

from dataclasses import dataclass, field

from .Layer import FixedPoint

from .Layer import LayerBase, Layer, Layer3D
from .MultiPortLayer import MultiPortLayerBase, MultiPortLayer, MultiPortLayer3D

from .BatchNormLayer import BatchNormLayer
from .InnerProductLayer import InnerProductLayer
from .PoolingLayer import PoolingLayer
from .HardswishLayer import HardswishLayer
from .ReLULayer import ReLULayer
from .ConvolutionLayer import ConvolutionLayer
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
from .ConvolutionLayer3D import ConvolutionLayer3D
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
