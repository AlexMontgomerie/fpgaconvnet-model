from enum import Enum
import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2

# Get enumeration from:
#   https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
class LAYER_TYPE(Enum):
    Concat       =3
    Convolution  =4
    Dropout      =6
    InnerProduct =14
    LRN          =15
    Pooling      =17
    ReLU         =18
    Sigmoid      =19
    Softmax      =20
    Split        =22
    Eltwise      =25
    # Not Enumerated
    BatchNorm = 40
    Scale     = 41
    Merge     = 42
    Squeeze   = 43
    Transpose = 44
    Flatten   = 45
    Cast      = 46
    Clip      = 47
    Shape     = 48
    AveragePooling = 49
    SiLU = 50 # i.e. Swish
    Reshape = 51
    NOP     = 52

    @classmethod
    def get_type(cls, t):
        if type(t) is str:
            return cls[t]
        elif type(t) is int:
            return cls(t)

def to_proto_layer_type(layer_type):
    layer_types = {
        LAYER_TYPE.Convolution      : fpgaconvnet_pb2.layer.layer_type.CONVOLUTION,
        LAYER_TYPE.InnerProduct     : fpgaconvnet_pb2.layer.layer_type.INNER_PRODUCT,
        LAYER_TYPE.Pooling          : fpgaconvnet_pb2.layer.layer_type.POOLING,
        LAYER_TYPE.AveragePooling   : fpgaconvnet_pb2.layer.layer_type.AVERAGE_POOLING,
        LAYER_TYPE.ReLU             : fpgaconvnet_pb2.layer.layer_type.ACTIVATION,
        LAYER_TYPE.Sigmoid          : fpgaconvnet_pb2.layer.layer_type.ACTIVATION,
        LAYER_TYPE.SiLU             : fpgaconvnet_pb2.layer.layer_type.ACTIVATION,
        LAYER_TYPE.Squeeze          : fpgaconvnet_pb2.layer.layer_type.SQUEEZE,
        LAYER_TYPE.Concat           : fpgaconvnet_pb2.layer.layer_type.CONCAT,
        LAYER_TYPE.BatchNorm        : fpgaconvnet_pb2.layer.layer_type.BATCH_NORM,
        LAYER_TYPE.Split            : fpgaconvnet_pb2.layer.layer_type.SPLIT,
        LAYER_TYPE.Eltwise          : fpgaconvnet_pb2.layer.layer_type.ELTWISE,
        LAYER_TYPE.NOP              : fpgaconvnet_pb2.layer.layer_type.SQUEEZE
    }
    return layer_types.get(layer_type, lambda: "Invalid Layer Type")

def from_proto_layer_type(layer_type):
    layer_types = {
        fpgaconvnet_pb2.layer.layer_type.CONVOLUTION        : LAYER_TYPE.Convolution,
        fpgaconvnet_pb2.layer.layer_type.INNER_PRODUCT      : LAYER_TYPE.InnerProduct,
        fpgaconvnet_pb2.layer.layer_type.POOLING            : LAYER_TYPE.Pooling,
        fpgaconvnet_pb2.layer.layer_type.AVERAGE_POOLING    : LAYER_TYPE.AveragePooling,
        fpgaconvnet_pb2.layer.layer_type.ACTIVATION         : [LAYER_TYPE.ReLU, LAYER_TYPE.Sigmoid, LAYER_TYPE.SiLU],
        fpgaconvnet_pb2.layer.layer_type.SQUEEZE            : LAYER_TYPE.Squeeze,
        fpgaconvnet_pb2.layer.layer_type.CONCAT             : LAYER_TYPE.Concat,
        fpgaconvnet_pb2.layer.layer_type.BATCH_NORM         : LAYER_TYPE.BatchNorm,
        fpgaconvnet_pb2.layer.layer_type.SPLIT              : LAYER_TYPE.Split,
        fpgaconvnet_pb2.layer.layer_type.ELTWISE            : LAYER_TYPE.Eltwise
    }
    return layer_types.get(layer_type, lambda: "Invalid Layer Type")

def from_onnx_op_type(op_type):
    layer_types = {
        # operations
        "Conv" : LAYER_TYPE.Convolution,
        "Gemm" : LAYER_TYPE.InnerProduct,
        "MatMul" : LAYER_TYPE.InnerProduct,
        "MaxPool" : LAYER_TYPE.Pooling,
        "AveragePool" : LAYER_TYPE.AveragePooling,
        "BatchNormalization" : LAYER_TYPE.BatchNorm,
        "LRN" : LAYER_TYPE.LRN,
        "GlobalAveragePool" : LAYER_TYPE.AveragePooling,
        "GlobalMaxPool" : LAYER_TYPE.AveragePooling, # TODO
        # branching nodes
        "Add" : LAYER_TYPE.EltWise,
        "Mul" : LAYER_TYPE.EltWise,
        "Concat" : LAYER_TYPE.Concat,
        # Activations
        "Relu" : LAYER_TYPE.ReLU,
        "Clip" : LAYER_TYPE.ReLU, # TODO: implement clip properly
        "Sigmoid" : LAYER_TYPE.Sigmoid, # TODO: implement clip properly
        "Softmax" : LAYER_TYPE.NOP, # TODO: move to CPU
        "Dropout" : LAYER_TYPE.Dropout,
        # shape operations
        "Transpose" : LAYER_TYPE.Transpose,
        "Squeeze" : LAYER_TYPE.Squeeze,
        "Cast" : LAYER_TYPE.Cast,
        "Flatten" : LAYER_TYPE.NOP, # NOTE: only "shape" layer supported
        "Reshape" : LAYER_TYPE.Reshape,
        "Shape" : LAYER_TYPE.Shape,
    }

    return layer_types.get(op_type, lambda: TypeError)
