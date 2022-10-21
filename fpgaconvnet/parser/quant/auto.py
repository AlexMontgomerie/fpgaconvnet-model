import math
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.parser.onnx.helper as onnx_helper

def quantise(graph, model, data_width=16, weight_width=16, acc_width=32):

    # iterate over nodes
    for node in graph.nodes:

        # special case for convolution and inner product
        if graph.nodes[node]["type"] in [ \
                LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:

            # get the max abs value from the weights
            weights = onnx_helper.get_model_initializer(
                    model, graph.nodes[node]["inputs"]["weights"])
            weights_max = np.amax(np.absolute(weights))

            # get the weight binary point
            weight_binary_point = weight_width - max(1,
                    int(math.ceil(math.log(weights_max, 2)))+1)

            # get the accumulation binary point
            acc_binary_point = weight_binary_point + data_width//2

            # adjust data types
            graph.nodes[node]["hw"].input_t.width = data_width
            graph.nodes[node]["hw"].input_t.binary_point = data_width//2
            graph.nodes[node]["hw"].weight_t.width = weight_width
            graph.nodes[node]["hw"].weight_t.binary_point = weight_binary_point
            graph.nodes[node]["hw"].acc_t.width = acc_width
            graph.nodes[node]["hw"].acc_t.binary_point = acc_binary_point
            graph.nodes[node]["hw"].output_t.width = data_width
            graph.nodes[node]["hw"].output_t.binary_point = data_width//2

        # for all others, just set the data type
        else:

            # adjust data types
            graph.nodes[node]["hw"].data_t.width = data_width
            graph.nodes[node]["hw"].data_t.binary_point = data_width//2

