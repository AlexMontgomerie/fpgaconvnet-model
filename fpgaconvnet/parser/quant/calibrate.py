import math
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.parser.onnx.helper as onnx_helper


def get_quant_param(model, data_width=16, weight_width=16, acc_width=40, calibration_data={}):

    def get_int_bits(node):
        if node in calibration_data:
            neg_int_bits = math.ceil(math.log(abs(float(calibration_data[node]["min_val"])))) + 1
            pos_int_bits = math.ceil(math.log(float(calibration_data[node]["max_val"])))
            # return max(neg_int_bits, pos_int_bits) + 2
            return 8
        else:
            print(f"WARNING: node {node} not in calibration data")
            # return data_width//2
            return 0

    # dictionary of quantisation parameters
    quant_param = {}

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # get the formatted name for the node
        node_name = onnx_helper.format_onnx_name(node)
        attr = onnx_helper.format_attr(node.attribute)
        attr.setdefault("data_width", data_width)
        attr.setdefault("acc_width", acc_width)

        # set default quant param
        quant_param[node_name] = {
            "acc_t" : {
                "width" : attr["acc_width"],
                "binary_point": attr["acc_width"]//2,
            },
        }

        # get the input bits
        input_integer_bits = max([ get_int_bits(n) for n in node.input ])

        # update the binary point for the output
        quant_param[node_name]["input_t"] = {
            "width" : data_width,
            "binary_point": data_width - input_integer_bits,
        }

        # get the output bits
        output_integer_bits = max([ get_int_bits(n) for n in node.output ])

        # update the binary point for the output
        quant_param[node_name]["output_t"] = {
            "width" : data_width,
            "binary_point": data_width - output_integer_bits,
        }

        # set data type to same as input type
        quant_param[node_name]["data_t"] = quant_param[node_name]["input_t"]

        # special case for convolution and inner product
        if node.op_type in [ "Conv", "Gemm" ]:
            attr.setdefault("weight_width", weight_width)
            attr.setdefault("block_floating_point", False)

            # get the max abs value from the weights
            weights = onnx_helper.get_model_initializer(model, node.input[1])
            weights_max = np.amax(np.absolute(weights))

            # get the weight binary point
            weight_binary_point = attr["weight_width"] - max(1,
                    int(math.ceil(math.log(weights_max, 2)))+1)

            # get the accumulation binary point
            acc_binary_point = weight_binary_point + quant_param[node_name]["input_t"]["binary_point"]

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
