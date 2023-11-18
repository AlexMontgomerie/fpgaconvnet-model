import math

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

from .chisel import SqueezeChisel
from .hls import SqueezeHLS, SqueezeHLS3D
