import math

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnxsim import simplify

import fpgaconvnet.parser.onnx.helper as onnx_helper

ADD_QUANT_ATTR=False

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
        try:
            next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue
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

def eliminate_nop_pool(model):
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
        try:
            next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue
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
        try:
            next_next_node = next(filter(lambda x: x.input[0] == next_node.output[0], model.graph.node))
        except StopIteration:
            continue
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
        try:
            next_node = next(filter(lambda x: x.input[0] == model.graph.node[0].output[0], model.graph.node))
        except StopIteration:
            return model

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
        try:
            prev_node = next(filter(lambda x: x.output[0] == model.graph.node[-1].input[0], model.graph.node))
        except StopIteration:
            return model
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
        try:
            next_node = next(filter(lambda x: node.output[0] in x.input, model.graph.node))
        except StopIteration:
            continue
        next_node_input_idx = list(next_node.input).index(node.output[0])
        output_shape = onnx_helper.get_input_shape(model,
                next_node.input[next_node_input_idx])

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
        try:
            flatten_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue
        if flatten_node.op_type != "Flatten":
            continue

        # get the gemm node
        try:
            gemm_node = next(filter(lambda x: x.input[0] == flatten_node.output[0], model.graph.node))
        except StopIteration:
            continue
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
        model.graph.node[index].name = f"{model.graph.node[index].op_type}_{index}"

    # return the new model
    return model

def absorb_quantise(model):
    model_was_changed = True

    zero_point_scale_inputs = []

    while model_was_changed:   # todo: move model_was_changed to Parser.py?
        model_was_changed = False
        for index, node in enumerate(model.graph.node):
            tensor_raw = onnx_helper.get_model_initializer(model, node.input[0], to_tensor=False)

            if node.op_type == "DequantizeLinear":

                # get scale and zero point parameters
                scale = onnx_helper.get_model_initializer(model, node.input[1], to_tensor=True)
                zero_point = onnx_helper.get_model_initializer(model, node.input[2], to_tensor=True)
                scale = np.atleast_1d(scale)
                zero_point = np.atleast_1d(zero_point)

                if tensor_raw is None:
                    # activation
                    quan_obj = "input"
                else:

                    # chenge initializers to floating point
                    tensor = onnx.numpy_helper.to_array(tensor_raw)
                    tensor_index = list(model.graph.initializer).index(tensor_raw)
                    size = scale.shape + (1,) * (tensor.ndim - scale.ndim)
                    dequant_tensor = (tensor-zero_point.reshape(size))*scale.reshape(size)

                    # create a new tensor
                    new_tensor = onnx.helper.make_tensor(
                        name=node.input[0],
                        data_type=onnx.TensorProto.FLOAT,
                        dims=tensor.shape,
                        vals=dequant_tensor.flatten().tolist())

                    # create new value info
                    tensor_value_info = onnx_helper.get_model_input(model, node.input[0])
                    tensor_value_info_index = list(model.graph.input).index(tensor_value_info)
                    new_tensor_value_info = onnx.helper.make_tensor_value_info(
                            node.input[0],
                            onnx.TensorProto.FLOAT,
                            tensor.shape)

                    # remove the old initializer, and insert the new one
                    model.graph.initializer.remove(tensor_raw)
                    model.graph.initializer.insert(tensor_index, new_tensor)
                    model.graph.input.remove(tensor_value_info)
                    model.graph.input.insert(tensor_value_info_index, new_tensor_value_info)

                    if tensor.ndim == 1:
                        # bias
                        quan_obj = "bias"
                    else:
                        # weight
                        quan_obj = "weight"

                # remove the zero point and scale inputs (if in inputs)
                zero_point_scale_inputs.extend(node.input[1:])

                #  move quant node input to the next node
                for next_node, input_index in onnx_helper.find_consumers(model, node.output[0]):
                    next_node.input[input_index] = node.input[0]
                    model_was_changed = True

                # remove quant node
                model.graph.node.remove(node)

                if ADD_QUANT_ATTR:
                    scale_attr = onnx.helper.make_attribute("{}_scale".format(quan_obj), scale)
                    next_node.attribute.append(scale_attr)
                    zero_point_attr = onnx.helper.make_attribute("{}_zero_point".format(quan_obj), zero_point)
                    next_node.attribute.append(zero_point_attr)

            if node.op_type == "QuantizeLinear":
                assert tensor_raw is None

                try:
                    prev_node = next(filter(lambda x: x.output[0] == node.input[0], model.graph.node))
                except StopIteration:
                    continue

                # remove the zero point and scale inputs (if in inputs)
                zero_point_scale_inputs.extend(node.input[1:])

                prev_node.output[0] = node.output[0]
                model.graph.node.remove(node)
                model_was_changed = True

                quan_obj = "input"

                if ADD_QUANT_ATTR:
                    scale_attr = onnx.helper.make_attribute("{}_scale".format(quan_obj), scale)
                    prev_node.attribute.append(scale_attr)
                    zero_point_attr = onnx.helper.make_attribute("{}_zero_point".format(quan_obj), zero_point)
                    prev_node.attribute.append(zero_point_attr)

    # remove zero_point and scale nodes
    for input in list(set(zero_point_scale_inputs)):
        model.graph.input.remove(onnx_helper.get_model_input(model, input))
        model.graph.initializer.remove(onnx_helper.get_model_initializer(model, input, to_tensor=False))

    # change input and output types
    model.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    model.graph.output[-1].type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    # change all value info types
    for vi in model.graph.value_info:
        vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    # simplify model (again)
    model, _ = simplify(model)

    return model


def fuse_mul_add_into_bn(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a mul node
        if node.op_type != "Mul":
            continue


        # get the next node
        try:
            next_node = next(filter(lambda x: len(x.input) > 1 and x.input[1] == node.output[0], model.graph.node))
        except StopIteration:
            continue

        print(node)
        # check noext node is add
        if next_node.op_type != "Add":
            continue

        # get the channels
        constant = onnx_helper.get_model_initializer(
                model, node.input[1], to_tensor=False)
        if constant is None:
            continue
        channels = constant.dims[0]

        # create zero mean
        zero_mean = onnx.helper.make_tensor(
            name="_".join([node.input[0],"zero_mean"]),
            data_type=onnx.TensorProto.FLOAT,
            dims=(channels,),
            vals=np.zeros((channels), dtype=np.float32))
        zero_mean_vi = onnx.helper.make_tensor_value_info(
                zero_mean.name,
                onnx.TensorProto.FLOAT,
                [channels])

        # update the graph
        model.graph.initializer.insert(-1,zero_mean)
        model.graph.input.insert(-1, zero_mean_vi)

        # create ones var
        one_var = onnx.helper.make_tensor(
            name="_".join([node.input[0],"one_var"]),
            data_type=onnx.TensorProto.FLOAT,
            dims=(channels,),
            vals=np.ones((channels), dtype=np.float32))
        one_var_vi = onnx.helper.make_tensor_value_info(
                one_var.name,
                onnx.TensorProto.FLOAT,
                [channels])

        # update the graph
        model.graph.initializer.insert(-1, one_var)
        model.graph.input.insert(-1, one_var_vi)

        # create a batch norm layer
        new_node = onnx.helper.make_node(
            "BatchNormalization",
            name=node.name+"_bn",
            inputs=[node.input[0], node.input[1],
                next_node.input[1], zero_mean_vi.name,
                one_var_vi.name], # TODO: add zero mean and var values
            outputs=next_node.output,
        )

        # remove old nodes and add new one
        model.graph.node.remove(node)
        model.graph.node.remove(next_node)
        model.graph.node.insert(index, new_node)

    return model

def fuse_bn_into_gemm(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a mul node
        if node.op_type != "Gemm":
            continue

        # get the next node
        try:
            next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue

        # check noext node is add
        if next_node.op_type != "BatchNormalization":
            continue

        # get B and C from GEMM
        B_raw = onnx_helper.get_model_initializer(model, node.input[1], to_tensor=False)
        C_raw = onnx_helper.get_model_initializer(model, node.input[2], to_tensor=False)

        # get the scale and bias terms from BN
        scale_raw = onnx_helper.get_model_initializer(model,
                next_node.input[1], to_tensor=False)
        bias_raw = onnx_helper.get_model_initializer(model,
                next_node.input[2], to_tensor=False)

        # get the new GEMM values
        B_new_val = np.multiply(onnx.numpy_helper.to_array(scale_raw),
                onnx.numpy_helper.to_array(B_raw))
        C_new_val = np.multiply(onnx.numpy_helper.to_array(scale_raw),
                onnx.numpy_helper.to_array(C_raw)) + onnx.numpy_helper.to_array(bias_raw)

        # update the initializer values for B
        B_index = list(model.graph.initializer).index(B_raw)
        new_B = onnx.helper.make_tensor(
            name=node.input[1],
            data_type=onnx.TensorProto.FLOAT,
            dims=B_new_val.shape, vals=B_new_val)
        # update weight's value info
        B_vi = onnx_helper.get_model_input(model, node.input[1])
        new_B_vi = onnx.helper.make_tensor_value_info(
                node.input[1],
                onnx.TensorProto.FLOAT,
                B_new_val.shape)
        # update the graph
        model.graph.initializer.remove(B_raw)
        model.graph.initializer.insert(B_index, new_B)
        model.graph.input.remove(B_vi)
        model.graph.input.insert(1, new_B_vi)

        # update the initializer values for C
        C_index = list(model.graph.initializer).index(C_raw)
        new_C = onnx.helper.make_tensor(
            name=node.input[2],
            data_type=onnx.TensorProto.FLOAT,
            dims=C_new_val.shape, vals=C_new_val)
        # update weight's value info
        C_vi = onnx_helper.get_model_input(model, node.input[2])
        new_C_vi = onnx.helper.make_tensor_value_info(
                node.input[2],
                onnx.TensorProto.FLOAT,
                C_new_val.shape)
        # update the graph
        model.graph.initializer.remove(C_raw)
        model.graph.initializer.insert(C_index, new_C)
        model.graph.input.remove(C_vi)
        model.graph.input.insert(2, new_C_vi)

        # update the output for Gemm
        node.output[0] = next_node.output[0]

        # remove old nodes and add new one
        model.graph.node.remove(next_node)

    return model

def remove_quant_nodes(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a quant node
        if node.op_type not in ["DequantizeLinear", "QuantizeLinear"]:
            continue

        # remove dequantize node
        if node.op_type == "DequantizeLinear":

            # get the next node
            try:
                next_node = next(filter(lambda x: node.output[0] in x.input, model.graph.node))
            except StopIteration:
                continue
            input_idx = list(next_node.input).index(node.output[0])

            # remove dequant node
            model.graph.node.remove(node)

            # copy input over
            next_node.input.remove(node.output[0])
            next_node.input.insert(input_idx, node.input[0])

            return remove_quant_nodes(model)

        # remove quantize node
        if node.op_type == "QuantizeLinear":

            # get the next node
            try:
                prev_node = next(filter(lambda x: node.input[0] in x.output, model.graph.node))
            except StopIteration:
                continue
            output_idx = list(prev_node.output).index(node.input[0])

            # remove dequant node
            model.graph.node.remove(node)

            # copy input over
            prev_node.output.remove(node.input[0])
            prev_node.output.insert(output_idx, node.output[0])

            return remove_quant_nodes(model)

    return model

def fuse_matmul_add_into_gemm(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a mul node
        if node.op_type != "MatMul":
            continue

        # get the next node
        try:
            next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue

        # check noext node is add
        if next_node.op_type != "Add":
            continue

        # create a Gemm node from these initialisers
        new_node = onnx.helper.make_node(
            "Gemm",
            name=node.name,
            inputs=[*node.input, next_node.input[1]],
            outputs=next_node.output,
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0
        )

        # remove old nodes and add new one
        model.graph.node.remove(node)
        model.graph.node.remove(next_node)
        model.graph.node.insert(index, new_node)

    return model

def fuse_mul_sigmoid_into_hardswish(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a mul node
        if node.op_type != "Mul":
            continue

        # check only two inputs
        if len(node.input) != 2:
            continue

        # check first mul input in graph
        if len(list(filter(lambda x: x.name == node.input[0], model.graph.input))):
            continue

        # check second mul input in graph
        if len(list(filter(lambda x: x.name == node.input[1], model.graph.input))):
            continue

        try:
            # get previous node a
            prev_node_a = next(filter(lambda x: x.output[0] == node.input[0], model.graph.node))
            # get previous node b
            prev_node_b = next(filter(lambda x: x.output[0] == node.input[1], model.graph.node))
        except StopIteration:
            continue

        # check prev node b is a sigmoid
        if prev_node_b.op_type != "Sigmoid":
            continue

        # check that the current node an prev node b have the same input
        if node.input[0] != prev_node_b.input[0]:
            continue

        # create a new HardSwish node
        new_node = onnx.helper.make_node(
            "HardSwish",
            name=node.name+"_hard_swish",
            inputs=[prev_node_a.output[0]],
            outputs=node.output,
        )

        # add the node to the graph
        model.graph.node.insert(index, new_node)

        # remove prev nodes
        model.graph.node.remove(node)
        model.graph.node.remove(prev_node_b)

    return model

def fuse_relu_into_previous(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a relu node
        if node.op_type != "Relu":
            continue

        try:
            prev_node = next(filter(lambda x: x.output[0] == node.input[0], model.graph.node))
        except StopIteration:
            continue
        next_nodes = filter(lambda x: (x.input[0] == node.output[0]) or (x.input[1] == node.output[0]) if len(x.input) > 1 else x.input[0] == node.output[0], model.graph.node)

        # Connect the previous node output to the next node(s) input
        for next_node in next_nodes:
            input_idx = list(next_node.input).index(node.output[0])
            next_node.input.remove(node.output[0])
            next_node.input.insert(input_idx, prev_node.output[0])

        # Remove the current node
        model.graph.node.remove(node)

    return model

def convert_to_version_15(model):
    return onnx.version_converter.convert_version(model, 14)


def convert_gemm_to_conv(model):
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a gemm node
        if node.op_type != "Gemm":
            continue

        # create a new conv node
        new_node = onnx.helper.make_node(
            "Conv",
            name=node.name,
            inputs=node.input,
            outputs=node.output,
            kernel_shape=[1,1,1],
        )

        # Remove the gemm node and add conv node
        model.graph.node.remove(node)
        model.graph.node.insert(index, new_node)

    return model

def eliminate_nop_pad(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a pad node
        if node.op_type != "Pad":
            continue

        # get the pad attribute
        pads = onnx_helper.get_model_initializer(model,
                node.input[1], to_tensor=True)

        # check if they are all zero
        if pads.any():
            continue

        # get the next node
        try:
            next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue

        # remove dropout node
        model.graph.node.remove(node)
        next_node.input[0] = node.input[0]

    return model

def move_relu_after_quant(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a pad node
        if node.op_type != "Relu":
            continue

        # get the next node
        try:
            next_node = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue

        # check next node is quantize linear
        if next_node.op_type != "QuantizeLinear":
            continue

        # get the index of next node
        next_index = list(model.graph.node).index(next_node)

        # get the next next node
        try:
            next_next_node = next(filter(lambda x: x.input[0] == next_node.output[0], model.graph.node))
        except StopIteration:
            continue

        # move the relu after quantize
        next_node.input[0] = node.input[0]
        node.input[0] = next_node.output[0]
        next_next_node.input[0] = node.output[0]

        # swap order of relu and dequant
        node = model.graph.node.pop(index)
        next_node = model.graph.node.pop(next_index-1)
        model.graph.node.insert(index, next_node)
        model.graph.node.insert(next_index, node)


        # find relu output value info
        node_vi = next(filter(lambda x: x.name == node.output[0], model.graph.value_info))
        next_node_vi = next(filter(lambda x: x.name == next_node.output[0], model.graph.value_info))

        # update the output type
        node_vi.type.tensor_type.elem_type = next_node_vi.type.tensor_type.elem_type

    return model

def insert_scale_shift_quant(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find Gemm and Conv nodes
        if node.op_type not in ["Gemm", "Conv"]:
            continue

        # get quant and dequant layers
        try:
            input_quant = next(filter(lambda x: \
                    x.output[0] == node.input[0], model.graph.node))
            weight_quant = next(filter(lambda x: \
                    x.output[0] == node.input[1], model.graph.node))
            output_quant = next(filter(lambda x: \
                    x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue

        # get input, weight and output scale
        input_scale = onnx_helper.get_model_initializer(model,
                        input_quant.input[1], to_tensor=True)
        weight_scale = np.atleast_1d(onnx_helper.get_model_initializer(model,
                        weight_quant.input[1], to_tensor=True))
        output_scale = onnx_helper.get_model_initializer(model,
                        output_quant.input[1], to_tensor=True)

        # get the per-channel scale and shift
        quant_scale = []
        quant_shift = []
        for ws in weight_scale:

            # get significand and exponent for output scale
            effective_os = input_scale * ws / output_scale
            man, exp = math.frexp(effective_os)

            # convert to quantised multiplication
            quant_scale.append(man * (1 << 31))
            quant_shift.append(31 - exp)

        # # add an empty bias term
        # quant_scale_init = onnx.helper.make_tensor(
        #     name="_".join([node.input[1],"bias"]),
        #     data_type=init.data_type,
        #     dims=(weights.shape[0],),
        #     vals=np.zeros(weights.shape[0]).flatten().tolist())
        # new_bias_value_info = onnx.helper.make_tensor_value_info(
        #         new_bias.name,
        #         onnx.TensorProto.FLOAT,
        #         [weights.shape[0]])
        #     # update the graph
        #     model.graph.initializer.insert(-1,new_bias)
        #     model.graph.input.insert(-1,new_bias_value_info)

        # # create batch norm node

        # print(quant_scale, quant_shift)

    return model

def fuse_add_clip_mul_div_into_hardswish(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # find a mul node
        if node.op_type != "Mul":
            continue

        # check only two inputs
        if len(node.input) != 2:
            continue

        # check first mul input in graph
        if len(list(filter(lambda x: x.name == node.input[0], model.graph.input))):
            continue

        # check second mul input in graph
        if len(list(filter(lambda x: x.name == node.input[1], model.graph.input))):
            continue

        try:
            # get previous node a
            prev_node_a = next(filter(lambda x: x.output[0] == node.input[0], model.graph.node))
            # get previous node b
            clip = next(filter(lambda x: x.output[0] == node.input[1], model.graph.node))
        except StopIteration:
            continue

        # check prev node b is a sigmoid
        if clip.op_type != "Clip":
            continue

        # get clip's previous node
        try:
            add = next(filter(lambda x: x.output[0] == clip.input[0], model.graph.node))
        except StopIteration:
            continue

        # check prev node b is a sigmoid
        if add.op_type != "Add":
            continue

        # check that the current node an prev node b have the same input
        if node.input[0] != add.input[0]:
            continue

        # get next node
        try:
            div = next(filter(lambda x: x.input[0] == node.output[0], model.graph.node))
        except StopIteration:
            continue

        # check prev node b is a sigmoid
        if div.op_type != "Div":
            continue

        # create a new HardSwish node
        new_node = onnx.helper.make_node(
            "HardSwish",
            name=node.name+"_hard_swish",
            inputs=[prev_node_a.output[0]],
            outputs=div.output,
        )

        # add the node to the graph
        model.graph.node.insert(index, new_node)

        # remove prev nodes
        model.graph.node.remove(node)
        model.graph.node.remove(clip)
        model.graph.node.remove(add)
        model.graph.node.remove(div)

    return model

def add_nop_to_split_output(model):

    # iterate over nodes in the graph
    for index, node in list(enumerate(model.graph.node)):

        # find a split node
        if node.op_type != "Split":
            continue

        # iterate over output nodes
        for i in range(len(node.output)):

            # create a new NOP (reshape with same shape)
            shape = onnx_helper.get_input_shape(model, node.output[i])
            shape_init_name = f"{node.name}_{i}_shape"
            shape_init = onnx.helper.make_tensor(
                name=shape_init_name,
                data_type=onnx.TensorProto.INT64,
                dims=[len(shape)],
                vals=shape)
            model.graph.initializer.insert(-1, shape_init)

            shape_init_value_info = onnx.helper.make_tensor_value_info(
                shape_init_name,
                onnx.TensorProto.INT64,
                [len(shape)])
            model.graph.input.insert(-1, shape_init_value_info)

            new_node = onnx.helper.make_node(
                "Reshape",
                name=f"{node.name}_{i}_nop",
                inputs=[node.output[i], shape_init_name],
                outputs=[node.output[i]+"_nop"],
            )

            # find the next nodes
            for next_node in list(filter(lambda x: \
                    node.output[i] in x.input, model.graph.node)):

                # get index of input
                input_idx = list(next_node.input).index(node.output[i])

                # update input
                next_node.input[input_idx] = node.output[i]+"_nop"

            # append node to graph
            model.graph.node.insert(index, new_node)

    # topological sort model
    g = gs.import_onnx(model)
    # g.cleanup()
    g.toposort()
    model = gs.export_onnx(g)

    return model

def remove_empty_inputs_outputs(model):

    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):

        # iterate over inputs and remove empty ones
        node.input[:] = list(filter(lambda x: x != "", node.input))

        # iterate over outputs and remove empty ones
        node.output[:] = list(filter(lambda x: x != "", node.output))

    return model