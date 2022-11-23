import math
import numpy as np
import onnx

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.parser.onnx.helper as onnx_helper
import fpgaconvnet.parser.onnx.passes as onnx_passes

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
                "width" : 8,
                "binary_point": 0,
            },
            "output_t" : {
                "width" : 8,
                "binary_point": 0,
            },
            "data_t" : {
                "width" : 8,
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
            continue

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
                    "scale": onnx.numpy_helper.to_array(scale).item(),
                    "zero_point": onnx.numpy_helper.to_array(zero_point).item(),
                }

        except StopIteration:
            continue

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
            quant_param[node_name] = {
                "weight_t" : {
                    "width" : weight_width,
                    "binary_point": 0,
                },
                "acc_t" : {
                    "width" : 32,
                    "binary_point": 0,
                },
            }

    # remove quantisation nodes
    onnx_passes.remove_quant_nodes(model)

    return quant_param


