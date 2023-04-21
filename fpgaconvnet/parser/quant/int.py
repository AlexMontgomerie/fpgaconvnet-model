import math
import numpy as np
import onnx

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.parser.onnx.helper as onnx_helper
import fpgaconvnet.parser.onnx.passes as onnx_passes

from fpgaconvnet.models.layers import BatchNormLayer

def get_quant_param(model):

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
                "width" : 32,
                "binary_point": 0,
            },
            "data_t" : {
                "width" : 32,
                "binary_point": 0,
            },
            "acc_t" : {
                "width" : 32,
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
                "width" : 32,
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

def get_scale_shift_node(quant_param, hw_node):

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
            man, exp = math.frexp(effective_os)

            # convert to quantised multiplication
            quant_scale.append(man * (1 << 31)) # TODO: handle edge cases, like https://github.com/tensorflow/tflite-micro/blob/365a9f3fbaa2fccd732315ac42ab6a13dff455cf/tensorflow/lite/kernels/internal/quantization_util.cc#L53-L105
            quant_shift.append(31 - exp)

        if len(quant_scale) == 1:

            # create a batch norm layer
            bn_node = BatchNormLayer(
                hw_node.hw.rows_out(),
                hw_node.hw.cols_out()*hw_node.hw.channels_out(),
                1
            )

        else:

            # create a batch norm layer
            bn_node = BatchNormLayer(
                hw_node.hw.rows_out(),
                hw_node.hw.cols_out(),
                hw_node.hw.channels_out()
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
        man, exp = math.frexp(effective_os)

        # convert to quantised multiplication
        quant_scale = [ man * (1 << 31) ]
        quant_shift = [ 31 - exp ]

        # create a batch norm layer
        bn_node = BatchNormLayer(
            hw_node.hw.rows_out(),
            hw_node.hw.cols_out()*hw_node.hw.channels_out(),
            1
        )

        # add scale and shift
        bn_node.scale = quant_scale
        bn_node.shift = quant_shift

        # return the batch norm layer
        return { "type": LAYER_TYPE.BatchNorm, "hw": bn_node, "onnx_node": hw_node.name }

