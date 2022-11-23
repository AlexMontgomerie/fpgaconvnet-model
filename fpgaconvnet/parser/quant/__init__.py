from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def quantise(graph, quant_format):


    # iterate over nodes
    for node in graph.nodes:

        # special case for convolution and inner product
        if graph.nodes[node]["type"] in [ \
                LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:

            # adjust data types
            graph.nodes[node]["hw"].input_t.width = quant_format[node]["input_t"]["width"]
            graph.nodes[node]["hw"].input_t.binary_point = quant_format[node]["input_t"]["binary_point"]
            graph.nodes[node]["hw"].weight_t.width = quant_format[node]["weight_t"]["width"]
            graph.nodes[node]["hw"].weight_t.binary_point = quant_format[node]["weight_t"]["binary_point"]
            graph.nodes[node]["hw"].acc_t.width = quant_format[node]["acc_t"]["width"]
            graph.nodes[node]["hw"].acc_t.binary_point = quant_format[node]["acc_t"]["binary_point"]
            graph.nodes[node]["hw"].output_t.width = quant_format[node]["output_t"]["width"]
            graph.nodes[node]["hw"].output_t.binary_point = quant_format[node]["output_t"]["binary_point"]

        # for all others, just set the data type
        else:

            # adjust data types
            graph.nodes[node]["hw"].data_t.width = quant_format[node]["data_t"]["width"]
            graph.nodes[node]["hw"].data_t.binary_point = quant_format[node]["data_t"]["binary_point"]

