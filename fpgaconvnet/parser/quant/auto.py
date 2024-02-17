import math
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.parser.onnx.helper as onnx_helper

def get_quant_param(model, data_width=16, weight_width=16,
        acc_width=30, block_floating_point=False, binary_point_scale=0.5):

    # dictionary of quantisation parameters
    quant_param = {}

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # get the formatted name for the node
        node_name = onnx_helper.format_onnx_name(node)
        attr = onnx_helper.format_attr(node.attribute)
        attr.setdefault("data_width", data_width)
        attr.setdefault("acc_width", acc_width)

        # default quant param
        quant_param[node_name] = {
            "input_t" : {
                "width" : attr["data_width"],
                "binary_point": int(attr["data_width"]*binary_point_scale),
            },
            "output_t" : {
                "width" : attr["data_width"],
                "binary_point": int(attr["data_width"]*binary_point_scale),
            },
            "data_t" : {
                "width" : attr["data_width"],
                "binary_point": int(attr["data_width"]*binary_point_scale),
            },
            "acc_t" : {
                "width" : attr["acc_width"],
                "binary_point": int(attr["acc_width"]*binary_point_scale),
            },
        }

        # special case for convolution and inner product
        if node.op_type in [ "Conv", "Gemm" ]:
            attr.setdefault("weight_width", weight_width)
            attr.setdefault("block_floating_point", block_floating_point)

            # get the max abs value from the weights
            weights = onnx_helper.get_model_initializer(model, node.input[1])
            weights_max = np.amax(np.absolute(weights))

            # get the weight binary point
            weight_binary_point = attr["weight_width"] - min(attr["weight_width"],
                    int(math.ceil(math.log(weights_max, 2)))+1)

            # get the accumulation binary point
            acc_binary_point = weight_binary_point + int(attr["data_width"]*binary_point_scale)

            # adjust data types
            quant_param[node_name]["weight_t"] = {
                "width" : attr["weight_width"],
                "binary_point": weight_binary_point,
            }
            quant_param[node_name]["acc_t"] = {
                "width" : attr["acc_width"],
                "binary_point": acc_binary_point,
            }
            quant_param[node_name]["block_floating_point"] = attr["block_floating_point"]

    # return the quant format
    return quant_param
