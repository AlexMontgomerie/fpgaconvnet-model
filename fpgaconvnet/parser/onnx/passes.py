import onnx
import numpy as np

import fpgaconvnet.parser.onnx.helper as onnx_helper

def convert_matmul_to_gemm(model):
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

def convert_pool_to_global_pool(model):
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):
        if node.op_type == "AveragePool":
            pass
    # return the new model
    return model

def remove_transpose_reshape(model):
    # iterate over nodes in the graph
    for index, node in enumerate(model.graph.node):
        if node.op_type not in [ "Conv", "ReLU", "MaxPool" ]:
            continue

        # get the next three nodes
        next_nodes = []
        try:
            next_nodes.append(next(filter(lambda x: x.input[0] == node.output[0], model.graph.node)))
            next_nodes.append(next(filter(lambda x: x.input[0] == next_nodes[0].output[0], model.graph.node)))
            next_nodes.append(next(filter(lambda x: x.input[0] == next_nodes[1].output[0], model.graph.node)))
        except StopIteration:
            continue

        # check they are the right nodes
        if next_nodes[0].op_type != "Transpose" or next_nodes[1].op_type != "Reshape" or next_nodes[2].op_type != "Gemm":
            continue

        # check the attributes of the transpose
        if next_nodes[0].attribute[0].ints != [0, 2, 3, 1]:
            continue

        # check reshape input
        reshape_shape = onnx_helper.get_model_initializer(model, next_nodes[1].input[1])
        if reshape_shape[0] != -1 or reshape_shape.shape != (2,):
            continue

        # finally, remove transpose and reshape node
        model.graph.node.remove(next_nodes[0])
        model.graph.node.remove(next_nodes[1])

        # connect node and Gemm node together
        next_nodes[-1].input[0] = node.output[0]

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


