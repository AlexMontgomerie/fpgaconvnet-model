import math
import inspect
from functools import reduce
from fpgaconvnet.tools.resource_analytical_model import bram_array_resource_model, uram_array_resource_model

def balance_module_rates(rate_graph):

    rate_ratio = [ abs(rate_graph[i,i+1]/rate_graph[i,i]) for i in range(rate_graph.shape[0]) ]

    for i in range(1,rate_graph.shape[0]):
        # start from end
        layer = rate_graph.shape[0]-i

        if abs(rate_graph[layer,layer]) > abs(rate_graph[layer-1,layer]):
            # propogate forward
            for j in range(layer,rate_graph.shape[0]):
                    if(abs(rate_graph[j,j]) <= abs(rate_graph[j-1,j])):
                        break
                    rate_graph[j,j]   = abs(rate_graph[j-1,j])
                    rate_graph[j,j+1] = -rate_graph[j,j]*rate_ratio[j]

        elif abs(rate_graph[layer,layer]) < abs(rate_graph[layer-1,layer]):
            # propogate backward
            for j in range(0,layer):
                    if(abs(rate_graph[layer-j,layer-j]) >= abs(rate_graph[layer-j-1,layer-j])):
                        break
                    rate_graph[layer-j-1,layer-j]   = -abs(rate_graph[layer-j,layer-j])
                    rate_graph[layer-j-1,layer-j-1] = -rate_graph[layer-j-1,layer-j]/rate_ratio[layer-j-1]
    return rate_graph

def get_factors(n):
    return sorted(list(set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))))

def stream_unit(self):
    '''
    unit width of streamed weights
    '''
    if self.weight_array_unit_width == 0:
        return 1
    unit = self.weight_array_num * math.ceil(self.weight_array_width/self.weight_array_num/self.weight_array_unit_width)
    return unit

def stream_step(self, level):
    '''
    level: 0 ~ 1, percentage of weights streamed in each optimization step
    return: number of RAM streamed in each optimization step
    '''
    if self.weight_array_unit_depth == 0:
        return 1
    step = math.ceil(level * math.ceil(self.weight_array_depth/self.weight_array_unit_depth))
    return step

def off_chip_addr_range(self):
    return min(self.weight_array_depth, (self.stream_weights / self.stream_unit()) * self.weight_array_unit_depth)

def on_chip_addr_range(self):
    return self.weight_array_depth - self.off_chip_addr_range()

def off_chip_buffer_size(self):
    return self.weight_array_unit_depth

def stream_bits(self):
    off_chip_bits = self.off_chip_addr_range() * self.weight_array_num * self.weight_array_width
    return off_chip_bits

def stream_cycles(self):
    cycles = self.on_chip_addr_range() + self.weight_array_unit_depth
    return cycles

def stream_bw(self):
    if self.stream_weights == 0:
        return 0
    else:
        return self.stream_bits() / self.stream_cycles()

def stream_buffer(self):
    if self.use_uram:
        return uram_array_resource_model(self.weight_array_unit_depth, self.weight_array_width/self.weight_array_num) * self.weight_array_num
    else:
        return bram_array_resource_model(self.weight_array_unit_depth, self.weight_array_width/self.weight_array_num, "memory") * self.weight_array_num

def stream_rsc(self, weight_array_depth, weight_array_width, weight_array_num): # todo: add extra logic cost
    self.weight_array_depth = weight_array_depth
    self.weight_array_width = weight_array_width * weight_array_num
    self.weight_array_num = weight_array_num

    if self.use_uram:
        weights_uram_usage = uram_array_resource_model(weight_array_depth, weight_array_width) * weight_array_num
        weights_uram_usage -= self.stream_weights
        self.weights_ram_usage = weights_uram_usage
        weights_bram_usage = 0
        if weights_uram_usage + self.stream_weights > 0:
            uram_details = uram_array_resource_model(weight_array_depth, weight_array_width, detailed=True)
            self.weight_array_unit_depth = uram_details[3]
            self.weight_array_unit_width = uram_details[1]
            if self.stream_weights > 0:
                weights_uram_usage += self.stream_buffer()
    else:
        weights_bram_usage = bram_array_resource_model(weight_array_depth, weight_array_width, "memory") * weight_array_num
        weights_bram_usage -= self.stream_weights
        self.weights_ram_usage = weights_bram_usage
        weights_uram_usage = 0
        if weights_bram_usage + self.stream_weights > 0:
            bram_details = bram_array_resource_model(weight_array_depth, weight_array_width, "memory", detailed=True)
            self.weight_array_unit_depth = bram_details[3]
            self.weight_array_unit_width = bram_details[1]
            if self.stream_weights > 0:
                weights_bram_usage += self.stream_buffer()

    # In cases where the depth and width are small enough resulting in no BRAMs or URAMs used the assert will fail because the self.weight_array_unit_depth and self.weight_array_unit_width are not set. In this case the assert is not needed because the weights_ram_usage is 0.
    if self.weights_ram_usage + self.stream_weights > 0:
        assert self.weights_ram_usage + self.stream_weights == \
            math.ceil(weight_array_depth/self.weight_array_unit_depth) \
            * math.ceil(weight_array_width/self.weight_array_unit_width) \
            * weight_array_num

    return weights_bram_usage, weights_uram_usage

def encode_rsc(node, encode_type):
    node_base_type = inspect.getmro(type(node))[-2]

    if encode_type == "none":
        return {}
    elif encode_type == "huffman":
        encode_rsc = {"LUT": 11, "FF": 7}
        decode_rsc = {"LUT": 400, "FF": 81}
    elif encode_type == "rle":
        encode_rsc = {"LUT": 26, "FF": 10}
        decode_rsc = {"LUT": 34, "FF": 11}

    rsc = {"LUT": 0, "FF": 0}
    for i, flag in enumerate(node.stream_inputs):
        if flag and node.input_compression_ratio[i] < 1:
            if node_base_type.__name__ in [ "Layer", "Layer3D" ]:
                rsc["LUT"] += decode_rsc["LUT"] * node.streams_in()
                rsc["FF"] += decode_rsc["FF"] * node.streams_in()
            elif node_base_type.__name__ in [ "MultiPortLayer", "MultiPortLayer3D" ]:
                rsc["LUT"] += decode_rsc["LUT"] * node.streams_in(i)
                rsc["FF"] += decode_rsc["FF"] * node.streams_in(i)
            else:
                raise NotImplementedError(f"base type {node_base_type}")
    for i, flag in enumerate(node.stream_outputs):
        if flag and node.output_compression_ratio[i] < 1:
            if node_base_type.__name__ in [ "Layer", "Layer3D" ]:
                rsc["LUT"] += encode_rsc["LUT"] * node.streams_out()
                rsc["FF"] += encode_rsc["FF"] * node.streams_out()
            elif node_base_type.__name__ in [ "MultiPortLayer", "MultiPortLayer3D" ]:
                rsc["LUT"] += encode_rsc["LUT"] * node.streams_out(i)
                rsc["FF"] += encode_rsc["FF"] * node.streams_out(i)
            else:
                raise NotImplementedError(f"base type {node_base_type}")
    if hasattr(node, "stream_weights") and node.stream_weights > 0 and node.weight_compression_ratio[0] < 1:
        rsc["LUT"] += decode_rsc["LUT"] * (node.weight_array_width // node.weight_t.width)
        rsc["FF"] += decode_rsc["FF"] * (node.weight_array_width // node.weight_t.width)
    return rsc
