import math
import numpy as np
import onnx

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.parser.onnx.helper as onnx_helper
import fpgaconvnet.parser.onnx.passes as onnx_passes

from fpgaconvnet.models.layers import BatchNormLayer

from fpgaconvnet.data_types import FixedPoint

def get_quant_param(model, acc_width = 24):

    # dictionary of quantisation parameters
    quant_param = {}

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # get the formatted name for the node
        node_name = onnx_helper.format_onnx_name(node)

        # skip all quantisation nodes
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            continue

        # default quant param
        quant_param[node_name] = {
            "input_t" : {
                "width" : 9,
                "binary_point": 0,
            },
            "output_t" : {
                "width" : acc_width,
                "binary_point": 0,
            },
            "data_t" : {
                "width" : acc_width,
                "binary_point": 0,
            },
            "acc_t" : {
                "width" : acc_width,
                "binary_point": 0,
            },
        }

        try:

            # get the previous node
            prev_node = next(filter(lambda x: x.output[0] == node.input[0], model.graph.node))

            # only if dequantize node
            if prev_node.op_type == "DequantizeLinear":

                # get scale and zero point
                scale = onnx_helper.get_model_initializer(model,
                        prev_node.input[1], to_tensor=False)
                zero_point = onnx_helper.get_model_initializer(model,
                        prev_node.input[2], to_tensor=False)

                # update quant parameters
                quant_param[node_name]["input_quant"] = {
                    "scale": onnx.numpy_helper.to_array(scale).item(),
                    "zero_point": onnx.numpy_helper.to_array(zero_point).item(),
                }

        except StopIteration:
            pass

        try:

            # get next nodes
            next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))

            # only if quantize node
            if next_node.op_type == "QuantizeLinear":

                # get scale and zero point
                scale = onnx_helper.get_model_initializer(model,
                        next_node.input[1], to_tensor=False)
                zero_point = onnx_helper.get_model_initializer(model,
                        next_node.input[2], to_tensor=False)

                # update quant parameters
                quant_param[node_name]["output_quant"] = {
                    "scale": onnx.numpy_helper.to_array(scale),
                    "zero_point": onnx.numpy_helper.to_array(zero_point),
                }

        except StopIteration:
            pass

        # special case for convolution and inner product
        if node.op_type in [ "Conv", "Gemm" ]:


            # get weight node
            weight_node = next(filter(lambda x: x.output[0] == node.input[1], model.graph.node))

            # get the minimum weight width
            weights = onnx_helper.get_model_initializer(model,
                    weight_node.input[0], to_tensor=True)
            weights_max = np.amax(np.absolute(weights))
            weight_width = int(math.ceil(math.log(weights_max, 2)))+1

            # update the quantisation parameters
            quant_param[node_name]["weight_t"] = {
                "width" : weight_width,
                "binary_point": 0,
            }
            quant_param[node_name]["acc_t"] = {
                "width" : acc_width,
                "binary_point": 0,
            }

            # get the scale and zero points for weights
            scale = onnx_helper.get_model_initializer(model,
                    weight_node.input[1], to_tensor=False)
            zero_point = onnx_helper.get_model_initializer(model,
                    weight_node.input[2], to_tensor=False)

            # update quant parameters
            quant_param[node_name]["weight_quant"] = {
                "scale": onnx.numpy_helper.to_array(scale),
                "zero_point": onnx.numpy_helper.to_array(zero_point),
            }


    # return quantisation parameters
    return quant_param

def quantise_mul(q, scale_width=32):

    # if zero, return zero
    if q == 0.0:
        return 0, 0

    # get the mantissa and exponent
    man, exp = math.frexp(q)
    man = int(man * (1 << scale_width-1))

    # scale down mantissa if it's the max value
    if man == ( 1 << (scale_width-1) ):
        exp += 1
        man /= 2

    # can't shift below -scale_width
    if exp < -(scale_width-1):
        exp = 0
        man = 0

    # saturate shift past scale_width
    if exp > scale_width-2:
        exp = scale_width-2
        man = ( 1 << (scale_width-1) ) - 1

    # return the mantissa and exponent
    return man, exp

def get_scale_shift_node(quant_param, hw_node, scale_width=24):

    if hw_node.layer_type in [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:

        # get the scale and shift
        input_scale  = quant_param["input_quant"]["scale"]
        weight_scale = np.atleast_1d(quant_param["weight_quant"]["scale"])
        output_scale = quant_param["output_quant"]["scale"]

        # get the per-channel scale and shift
        quant_scale = []
        quant_shift = []
        for ws in weight_scale:

            # get significand and exponent for output scale
            effective_os = input_scale * ws / output_scale
            man, exp = quantise_mul(effective_os, scale_width=scale_width)

            # convert to quantised multiplication
            quant_scale.append(man)
            quant_shift.append((scale_width-1) - exp)

        if len(quant_scale) == 1:

            # create a batch norm layer
            bn_node = BatchNormLayer(
                hw_node.hw.rows_out(),
                hw_node.hw.cols_out()*hw_node.hw.channels_out(), 1,
                input_t = hw_node.hw.output_t,
                scale_t = FixedPoint(scale_width, 0),
            )

        else:

            # create a batch norm layer
            bn_node = BatchNormLayer(
                hw_node.hw.rows_out(),
                hw_node.hw.cols_out(),
                hw_node.hw.channels_out(),
                input_t = hw_node.hw.output_t,
                scale_t = FixedPoint(scale_width, 0)
            )

        # add scale and shift
        bn_node.scale = quant_scale
        bn_node.shift = quant_shift

        # return the batch norm layer
        return { "type": LAYER_TYPE.BatchNorm, "hw": bn_node, "onnx_node": hw_node.name }

    elif hw_node.layer_type in [ LAYER_TYPE.GlobalPooling ]:

        # get the scale and shift
        input_scale  = quant_param["input_quant"]["scale"]
        output_scale = quant_param["output_quant"]["scale"]

        # get significand and exponent for output scale
        effective_os = input_scale / output_scale
        man, exp = quantise_mul(effective_os, scale_width=scale_width)

        # convert to quantised multiplication
        quant_scale = [ man ]
        quant_shift = [ (scale_width-1) - exp ]

        # create a batch norm layer
        bn_node = BatchNormLayer(
            hw_node.hw.rows_out(),
            hw_node.hw.cols_out()*hw_node.hw.channels_out(), 1,
            input_t = hw_node.hw.data_t,
            scale_t = FixedPoint(scale_width, 0)

        )

        # add scale and shift
        bn_node.scale = quant_scale
        bn_node.shift = quant_shift

        # return the batch norm layer
        return { "type": LAYER_TYPE.BatchNorm, "hw": bn_node, "onnx_node": hw_node.name }

