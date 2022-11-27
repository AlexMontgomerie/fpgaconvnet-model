import math
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.parser.onnx.helper as onnx_helper

def get_quant_param(model, data_width=16, weight_width=16, acc_width=32):

    # dictionary of quantisation parameters
    quant_param = {}

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # get the formatted name for the node
        node_name = onnx_helper.format_onnx_name(node)

        # default quant param
        quant_param[node_name] = {
            "input_t" : {
                "width" : data_width,
                "binary_point": data_width//2,
            },
            "output_t" : {
                "width" : data_width,
                "binary_point": data_width//2,
            },
            "data_t" : {
                "width" : data_width,
                "binary_point": data_width//2,
            },
        }

        # special case for convolution and inner product
        if node.op_type in [ "Conv", "Gemm" ]:

            # get the max abs value from the weights
            weights = onnx_helper.get_model_initializer(model, node.input[1])
            weights_max = np.amax(np.absolute(weights))

            # get the weight binary point
            weight_binary_point = weight_width - max(1,
                    int(math.ceil(math.log(weights_max, 2)))+1)

            # get the accumulation binary point
            acc_binary_point = weight_binary_point + data_width//2

            # adjust data types
            quant_param[node_name]["weight_t"] = {
                "width" : weight_width,
                "binary_point": weight_binary_point,
            }
            quant_param[node_name]["acc_t"] = {
                "width" : acc_width,
                "binary_point": acc_binary_point,
            }

    # return the quant format
    return quant_param
