from .architecture import BACKEND, DIMENSIONALITY, Architecture

# default architectures
CHISEL_2D_ARCH  = Architecture(backend=BACKEND.CHISEL, dimensionality=DIMENSIONALITY.TWO)
HLS_2D_ARCH     = Architecture(backend=BACKEND.HLS, dimensionality=DIMENSIONALITY.TWO)
CHISEL_3D_ARCH  = Architecture(backend=BACKEND.CHISEL, dimensionality=DIMENSIONALITY.THREE)
HLS_3D_ARCH     = Architecture(backend=BACKEND.HLS, dimensionality=DIMENSIONALITY.THREE)
