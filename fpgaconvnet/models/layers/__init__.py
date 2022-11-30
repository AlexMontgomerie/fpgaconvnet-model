"""
Layers are comprised of modules. They have the same functionality of the equivalent layers of the CNN model.
"""

from .Layer import FixedPoint

from .Layer import Layer
from .MultiPortLayer import MultiPortLayer

from .BatchNormLayer import BatchNormLayer
from .InnerProductLayer import InnerProductLayer
from .PoolingLayer import PoolingLayer
from .ReLULayer import ReLULayer
from .ConvolutionLayer import ConvolutionLayer
from .SqueezeLayer import SqueezeLayer
from .SplitLayer import SplitLayer
from .ConcatLayer import ConcatLayer
from .EltWiseLayer import EltWiseLayer
from .AveragePoolingLayer import AveragePoolingLayer

# 3D layers
from .Layer3D import Layer3D
from .MultiPortLayer3D import MultiPortLayer3D

# from .InnerProductLayer3D import InnerProductLayer3D
from .PoolingLayer3D import PoolingLayer3D
from .ActivationLayer3D import ActivationLayer3D
from .ReLULayer3D import ReLULayer3D
# from .ConvolutionLayer3D import ConvolutionLayer3D
from .SqueezeLayer3D import SqueezeLayer3D
# from .EltWiseLayer3D import EltWiseLayer3D
# from .AveragePoolingLayer3D import AveragePoolingLayer3D