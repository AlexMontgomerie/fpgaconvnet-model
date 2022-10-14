import onnx
import numpy as np

import fpgaconvnet.parser.onnx.helper as onnx_helper

def convert_matmul_to_gemm(model):
    """
    converts standalone matmul nodes to gemm nodes. This unifies the representation
    of InnerProduct layers
    """
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):
        if node.op_type == "MatMul":
            # update the weights
            init = onnx_helper.get_model_initializer(model, node.input[1], to_tensor=False)
            init_index = list(model.graph.initializer).index(init)
            weights = onnx.numpy_helper.to_array(init)
            weights = np.swapaxes(weights,0,1)
            new_init = onnx.helper.make_tensor(
                name=node.input[1],
                data_type=init.data_type,
                dims=weights.shape,
                vals=weights.flatten().tolist())
            # update weight's value info
            init_value_info = onnx_helper.get_model_input(model, node.input[1])
            init_value_info_index = list(model.graph.input).index(init_value_info)
            new_init_value_info = onnx.helper.make_tensor_value_info(
                    node.input[1],
                    onnx.TensorProto.FLOAT,
                    weights.shape)
            # update the graph
            model.graph.initializer.remove(init)
            model.graph.initializer.insert(init_index, new_init)
            model.graph.input.remove(init_value_info)
            model.graph.input.insert(init_value_info_index, new_init_value_info)
            # add an empty bias term
            new_bias = onnx.helper.make_tensor(
                name="_".join([node.input[1],"bias"]),
                data_type=init.data_type,
                dims=(weights.shape[0],),
                vals=np.zeros(weights.shape[0]).flatten().tolist())
            new_bias_value_info = onnx.helper.make_tensor_value_info(
                    new_bias.name,
                    onnx.TensorProto.FLOAT,
                    [weights.shape[0]])
            # update the graph
            model.graph.initializer.insert(-1,new_bias)
            model.graph.input.insert(-1,new_bias_value_info)
            # create a new matmul node
            new_node = onnx.helper.make_node(
                "Gemm",
                name=node.name,
                inputs=[*node.input, "_".join([node.input[1],"bias"])],
                outputs=node.output,
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=1
            )
            # remove old node and add new one
            model.graph.node.remove(node)
            model.graph.node.insert(index, new_node)
    # return the new model
    return model

def remove_transpose_reshape_to_gemm(model):
    """
    removes the transpose-reshape layers used to move from NCHW to NC for
    gemm layers. Results in re-ordering of the Gemm layer to compensate
    """

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Transpose":
            continue

        # get the next three nodes
        next_nodes = []
        try:
            next_nodes.append(next(filter(lambda x: x.input[0] == node.output[0], model.graph.node)))
            next_nodes.append(next(filter(lambda x: x.input[0] == next_nodes[0].output[0], model.graph.node)))
        except StopIteration:
            continue

        # check they are the right nodes
        if next_nodes[0].op_type != "Reshape" or next_nodes[1].op_type != "Gemm":
            continue

        # check the attributes of the transpose
        if node.attribute[0].ints != [0, 2, 3, 1]:
            continue

        # check reshape input
        reshape_shape = onnx_helper.get_model_initializer(model, next_nodes[0].input[1])
        if reshape_shape[0] != -1 or reshape_shape.shape != (2,):
            continue

        print(f"WARNING: removing transpose node {node.name} and rehape node {next_nodes[0].name} \
                as the NC ordering is implicit in hardware")
        print(f"CRITICAL WARNING: weights for gemm node are not re-ordered (WIP)")

        # finally, remove transpose and reshape node
        model.graph.node.remove(node)
        model.graph.node.remove(next_nodes[0])

        # connect node and Gemm node together
        next_nodes[1].input[0] = node.input[0]

    # return the new model
    return model

def remove_channel_first_transpose(model):
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find transpose nodes
        if node.op_type !=  "Transpose":
            continue

        # get the previous shape
        input_shape = onnx_helper.get_input_shape(model, node.input[0])

        # must be 3d featuremap
        if len(input_shape) != 4:
            continue

        # spatial dimensions must be 1
        if input_shape[2] != 1 or input_shape[3] != 1:
            continue

        # check the attributes of the transpose
        if node.attribute[0].ints != [0, 2, 3, 1]:
            continue

        # get the next node
        next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))

        # finally, remove transpose and reshape node
        model.graph.node.remove(node)

        # connect node and Gemm node together
        next_node.input[0] = node.input[0]

    # return the new model
    return model

def remove_channel_first_reshape(model):
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find transpose nodes
        if node.op_type !=  "Reshape":
            continue

        # get the previous shape
        input_shape = onnx_helper.get_input_shape(model, node.input[0])

        # must be 3d featuremap
        if len(input_shape) != 4:
            continue

        # spatial dimensions must be 1
        if input_shape[2] != 1 or input_shape[3] != 1:
            continue

        # check reshape input
        reshape_shape = onnx_helper.get_model_initializer(model, node.input[1])
        if reshape_shape[0] != -1 or reshape_shape.shape != (2,):
            continue

        # get the next node
        next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))

        # finally, remove transpose and reshape node
        model.graph.node.remove(node)

        # connect node and Gemm node together
        next_node.input[0] = node.input[0]

    # return the new model
    return model

def remove_redundant_flatten(model):
    """
    removes flatten layers where the input shape and output shape are exactly
    the same.
    """
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find transpose nodes
        if node.op_type !=  "Flatten":
            continue

        # get shapes in and out
        input_shape = onnx_helper.get_input_shape(model, node.input[0])
        output_shape = onnx_helper.get_input_shape(model, node.output[0])

        # see if input and output shape the same
        if input_shape != output_shape:
            continue

        print(f"WARNING: removing flatten node {node.name}, \
                as the input and output shapes are the same")

        # get the next node
        next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))

        # finally, remove flatten node
        model.graph.node.remove(node)
        next_node.input[0] = node.input[0]

    # return the new model
    return model

def remove_training_nodes(model):
    """
    removes nodes used for training (Dropout), as this is only an inference engine
    """
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find transpose nodes
        if node.op_type not in ["Dropout"]:
            continue

        print(f"WARNING: removing dropout node {node.name} \
                as it is not used for inference")

        # get the next node
        next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))

        # remove dropout node
        model.graph.node.remove(node)
        next_node.input[0] = node.input[0]

    # return the new model
    return model

def remove_flatten_to_gemm(model):
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a flatten node
        if node.op_type != "Flatten":
            continue

        # get the next three nodes
        next_node = None
        try:
            next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue

        # check if next node is a Gemm node
        if next_node.op_type != "Gemm":
            continue

        # TODO: reshape the Gemm initialiser!

        # finally, remove transpose and reshape node
        model.graph.node.remove(node)

        # connect node and Gemm node together
        next_node.input[0] = node.input[0]

    # return the new model
    return model

def remove_redundant_pooling(model):
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a flatten node
        if node.op_type not in ["MaxPool", "AveragePool"]:
            continue

        # check kernel size isn't 1
        attr = onnx_helper.format_attr(node.attribute)
        if np.prod(attr["kernel_shape"]) != 1:
            continue

        # finally, remove transpose and reshape node
        model.graph.node.remove(node)

        # connect node and Gemm node together
        next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        next_node.input[0] = node.input[0]

    # return the new model
    return model

def convert_pool_to_global_pool(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a flatten node
        if node.op_type not in ["MaxPool", "AveragePool"]:
            continue

        # check kernel size equals spatial dimension
        input_shape = onnx_helper.get_input_shape(model, node.input[0])
        kernel_shape = onnx_helper.format_attr(node.attribute)["kernel_shape"]
        if kernel_shape[0] != input_shape[-2] or kernel_shape[1] != input_shape[-1]:
            continue

        # convert to Global node
        if node.op_type == "MaxPool":
            model.graph.node[index].op_type = "GlobalMaxPool"
        if node.op_type == "AveragePool":
            model.graph.node[index].op_type = "GlobalAveragePool"

        # remove attributes
        del model.graph.node[index].attribute[:]

    # return the new model
    return model

def fuse_matmul_add_into_gemm(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a flatten node
        if node.op_type != "MatMul":
            continue

        # get the next node
        next_node = None
        try:
            next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue

        # check next node is add
        if next_node.op_type != "Add":
            continue

        # remove add and matmul node
        model.graph.node.remove(node)
        model.graph.node.remove(next_node)

        # create a Gemm node with the matmul weights and add bias
        new_node = onnx.helper.make_node(
            "Gemm",
            name=node.name,
            inputs=[*node.input, next_node.input[1]],
            outputs=node.output,
            alpha=1.0, beta=1.0,
            transA=0, transB=0
        )

        # add new one
        model.graph.node.insert(index, new_node)

        # connect node and Gemm node together
        next_next_node = next(filter(lambda x: x.input[0] == next_node.output[0], model.graph.node))
        next_next_node.input[0] = new_node.output[0]

    # return the new model
    return model

def remove_first_transpose(model):

    # check first node
    if model.graph.node[0].op_type == "Transpose":

        # # check transpose is moving channels back
        # if model.graph.node[0].ints != [0, 2, 3, 1]:
        #     return model

        print(f"WARNING: removing first transpose ({model.graph.node[0].name}), as fpgaConvNet is already channel-first")

        # get next node
        next_node = next(filter(lambda x: x.input[0] == model.graph.node[0].output[0], model.graph.node))

        # remove node
        next_node.input[0] = model.graph.node[0].input[0]
        model.graph.node.remove(model.graph.node[0])

        # return modified model
        return model

    # return un-modified model
    return model

def remove_last_softmax(model):

    # check first node
    if model.graph.node[-1].op_type == "Softmax":

        print(f"WARNING: removing last softmax node ({model.graph.node[-1].name}), must be implemented on the CPU instead")

        # get prev node
        prev_node = next(filter(lambda x: x.output[0] == model.graph.node[-1].input[0], model.graph.node))

        # remove node
        prev_node.output[0] = model.graph.node[-1].output[0]
        model.graph.node.remove(model.graph.node[-1])

        # return modified model
        return model

    # return un-modified model
    return model


def convert_reshape_to_flatten(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a flatten node
        if node.op_type != "Reshape":
            continue

        # get input and output shape
        input_shape = onnx_helper.get_input_shape(model, node.input[0])
        next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        output_shape = onnx_helper.get_input_shape(model, next_node.input[0])

        # check the output shape is the same as flattened
        if np.prod(input_shape[1:]) != output_shape[1]:
            continue

        # create a new Flaten node
        new_node = onnx.helper.make_node(
            "Flatten",
            name=node.name,
            inputs=[node.input[0]],
            outputs=node.output,
        )

        # remove old node and add new one
        model.graph.node.remove(node)
        model.graph.node.insert(index, new_node)

    # return the new model
    return model

def convert_transpose_flatten_gemm_to_flatten_gemm(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a flatten node
        if node.op_type != "Transpose":
            continue

        # get the flatten node
        flatten_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        if flatten_node.op_type != "Flatten":
            continue

        # get the gemm node
        gemm_node = next(filter(lambda x: x.input[0] == flatten_node.output[0], model.graph.node))
        if gemm_node.op_type != "Gemm":
            continue

        # get shape before and after transpose
        pre_transpose_shape = onnx_helper.get_input_shape(model, node.input[0])
        post_transpose_shape = onnx_helper.get_input_shape(model, node.output[0])

        # transpose shape
        trans = node.attribute[0].ints

        # get weights
        weights_raw = onnx_helper.get_model_initializer(model, gemm_node.input[1], to_tensor=False)
        weights_index = list(model.graph.initializer).index(weights_raw)
        weights_type = weights_raw.data_type

        # perform reshape-transpose-flatten on weights
        weights = onnx.numpy_helper.to_array(weights_raw)
        weights = np.reshape(weights, (-1, *pre_transpose_shape[1:]), order="F")
        weights = np.transpose(weights, (0, *trans[1:]))
        weights = np.reshape(weights, (-1, np.prod(post_transpose_shape[1:])), order="F")

        # update the weights
        new_weights = onnx.helper.make_tensor(
            name=gemm_node.input[1],
            data_type=weights_type,
            dims=weights.shape,
            vals=weights.flatten().tolist())

        # update weight's value info
        weights_value_info = onnx_helper.get_model_input(model, gemm_node.input[1])
        weights_value_info_index = list(model.graph.input).index(weights_value_info)
        new_weights_value_info = onnx.helper.make_tensor_value_info(
                gemm_node.input[1],
                onnx.TensorProto.FLOAT,
                weights.shape)

        # update the graph with modified weights
        model.graph.initializer.remove(weights_raw)
        model.graph.initializer.insert(weights_index, new_weights)
        model.graph.input.remove(weights_value_info)
        model.graph.input.insert(weights_value_info_index, new_weights_value_info)

        # remove transpose node
        model.graph.node.remove(node)
        flatten_node.input[0] = node.input[0]

    # return the new model
    return model

def make_clip_min_max_scalar(model): #TODO

    # # iterate over nodes in the graph
    # for index, node in enumerate(model.graph.node):

    #     # find a flatten node
    #     if node.op_type != "Clip":
    #         continue

    #     # update min value
    #     min_value = onnx_helper.get_model_initializer(model, node.input[1], to_tensor=False)
    #     print(min_value)
    #     if isinstance(min_value, list):
    #         print("here")

    #     print(node.input[1])
    #     print(node.input[2])

    # return the new model
    return model

def rename_all_nodes(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):
        model.graph.node[index].name = onnx_helper.format_onnx_name(node)

    # return the new model
    return model


