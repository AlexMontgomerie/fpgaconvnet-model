"""
Layers are comprised of modules. They have the same functionality of the equivalent layers of the CNN model.
"""

from .Layer             import Layer

from .BatchNormLayer    import BatchNormLayer
from .InnerProductLayer import InnerProductLayer
from .PoolingLayer      import PoolingLayer
from .ReLULayer         import ReLULayer
from .ConvolutionLayer  import ConvolutionLayerBase
from .SqueezeLayer      import SqueezeLayer
