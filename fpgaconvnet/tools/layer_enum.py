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
    LogSoftmax   =21
    Split        =22
    EltWise      =25
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
    GlobalPooling = 49
    HardSigmoid = 50
    HardSwish   = 51
    Reshape     = 52
    NOP         = 53
    LeakyReLU   = 54
    ReSize      = 55
    Chop        = 56
    Pad         = 57
    SiLU        = 58
    ThresholdedReLU = 59

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
        LAYER_TYPE.GlobalPooling    : fpgaconvnet_pb2.layer.layer_type.AVERAGE_POOLING,
        LAYER_TYPE.ReLU             : fpgaconvnet_pb2.layer.layer_type.RELU,
        LAYER_TYPE.Sigmoid          : fpgaconvnet_pb2.layer.layer_type.SIGMOID,
        LAYER_TYPE.HardSigmoid      : fpgaconvnet_pb2.layer.layer_type.HARD_SIGMOID,
        LAYER_TYPE.HardSwish        : fpgaconvnet_pb2.layer.layer_type.HARD_SWISH,
        LAYER_TYPE.SiLU             : fpgaconvnet_pb2.layer.layer_type.SILU,
        LAYER_TYPE.ThresholdedReLU  : fpgaconvnet_pb2.layer.layer_type.THRESHOLDEDRELU,
        LAYER_TYPE.Squeeze          : fpgaconvnet_pb2.layer.layer_type.SQUEEZE,
        LAYER_TYPE.Concat           : fpgaconvnet_pb2.layer.layer_type.CONCAT,
        LAYER_TYPE.BatchNorm        : fpgaconvnet_pb2.layer.layer_type.BATCH_NORM,
        LAYER_TYPE.Split            : fpgaconvnet_pb2.layer.layer_type.SPLIT,
        LAYER_TYPE.EltWise          : fpgaconvnet_pb2.layer.layer_type.ELTWISE,
        LAYER_TYPE.ReSize           : fpgaconvnet_pb2.layer.layer_type.RESIZE,
        LAYER_TYPE.Chop             : fpgaconvnet_pb2.layer.layer_type.CHOP,
        LAYER_TYPE.Reshape          : fpgaconvnet_pb2.layer.layer_type.SQUEEZE,
        LAYER_TYPE.NOP              : fpgaconvnet_pb2.layer.layer_type.SQUEEZE,
    }

    # get the layer type (protobuf)
    try:
        layer_type_pb = layer_types[layer_type]
    except KeyError:
        raise TypeError("Invalid Layer Type")

    return layer_type_pb
def from_proto_layer_type(layer_type):
    layer_types = {
        fpgaconvnet_pb2.layer.layer_type.CONVOLUTION        : LAYER_TYPE.Convolution,
        fpgaconvnet_pb2.layer.layer_type.INNER_PRODUCT      : LAYER_TYPE.InnerProduct,
        fpgaconvnet_pb2.layer.layer_type.POOLING            : LAYER_TYPE.Pooling,
        fpgaconvnet_pb2.layer.layer_type.AVERAGE_POOLING    : LAYER_TYPE.GlobalPooling,
        fpgaconvnet_pb2.layer.layer_type.RELU               : LAYER_TYPE.ReLU,
        fpgaconvnet_pb2.layer.layer_type.SILU               : LAYER_TYPE.SiLU,
        fpgaconvnet_pb2.layer.layer_type.SIGMOID            : LAYER_TYPE.Sigmoid,
        fpgaconvnet_pb2.layer.layer_type.HARD_SIGMOID       : LAYER_TYPE.HardSigmoid,
        fpgaconvnet_pb2.layer.layer_type.HARD_SWISH         : LAYER_TYPE.HardSwish,
        fpgaconvnet_pb2.layer.layer_type.RESIZE             : LAYER_TYPE.ReSize,
        fpgaconvnet_pb2.layer.layer_type.CHOP               : LAYER_TYPE.Chop,
        fpgaconvnet_pb2.layer.layer_type.THRESHOLDEDRELU    : LAYER_TYPE.ThresholdedReLU,
        fpgaconvnet_pb2.layer.layer_type.SQUEEZE            : LAYER_TYPE.Squeeze,
        fpgaconvnet_pb2.layer.layer_type.CONCAT             : LAYER_TYPE.Concat,
        fpgaconvnet_pb2.layer.layer_type.BATCH_NORM         : LAYER_TYPE.BatchNorm,
        fpgaconvnet_pb2.layer.layer_type.SPLIT              : LAYER_TYPE.Split,
        fpgaconvnet_pb2.layer.layer_type.ELTWISE            : LAYER_TYPE.EltWise,
        fpgaconvnet_pb2.layer.layer_type.RESIZE             : LAYER_TYPE.ReSize,
    }
    return layer_types.get(layer_type, lambda: "Invalid Layer Type")

def from_onnx_op_type(op_type):
    layer_types = {
        # operations
        "Conv" : LAYER_TYPE.Convolution,
        "Gemm" : LAYER_TYPE.InnerProduct,
        "MatMul" : LAYER_TYPE.InnerProduct,
        "MaxPool" : LAYER_TYPE.Pooling,
        "GlobalPool" : LAYER_TYPE.GlobalPooling,
        "BatchNormalization" : LAYER_TYPE.BatchNorm,
        "LRN" : LAYER_TYPE.LRN,
        "GlobalAveragePool" : LAYER_TYPE.GlobalPooling,
        "GlobalMaxPool" : LAYER_TYPE.GlobalPooling, # TODO
        # branching nodes
        "Add" : LAYER_TYPE.EltWise,
        "Mul" : LAYER_TYPE.EltWise,
        "Concat" : LAYER_TYPE.Concat,
        # Activations
        "LeakyRelu" : LAYER_TYPE.ReLU,
        "Relu" : LAYER_TYPE.ReLU,
        "Clip" : LAYER_TYPE.ReLU, # TODO: implement
        "Sigmoid" : LAYER_TYPE.Sigmoid, # TODO: implement
        "HardSigmoid" : LAYER_TYPE.HardSigmoid, # TODO: implement
        "HardSwish" : LAYER_TYPE.HardSwish, # TODO: implement
        "ThresholdedRelu" : LAYER_TYPE.ThresholdedReLU,
        "Softmax" : LAYER_TYPE.NOP, # TODO: move to CPU
        "LogSoftmax" : LAYER_TYPE.NOP, # TODO: move to CPU
        "Dropout" : LAYER_TYPE.Dropout,
        "Silu" : LAYER_TYPE.SiLU, # TODO: Silu does not exist as an onnx operator
        # shape operations
        "Transpose" : LAYER_TYPE.Transpose,
        "Squeeze" : LAYER_TYPE.Squeeze,
        "Cast" : LAYER_TYPE.Cast,
        "Flatten" : LAYER_TYPE.NOP, # NOTE: only "shape" layer supported
        "Reshape" : LAYER_TYPE.Reshape,
        "Shape" : LAYER_TYPE.Shape,
        "Resize" : LAYER_TYPE.ReSize,
        "Split" : LAYER_TYPE.Chop,
        "Pad" : LAYER_TYPE.NOP,
    }

    return layer_types.get(op_type, lambda: TypeError)

def from_cfg_type(op_type):
    if op_type == "*":
        return "*"
    elif op_type == "Split":
        return LAYER_TYPE.Split
    elif op_type == "Add" or op_type == "Mul":
        return op_type.lower()
    else:
        return from_onnx_op_type(op_type)
