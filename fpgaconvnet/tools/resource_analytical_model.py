import bisect
import math

BRAM_CONF_WIDTH={1:16384, 2:8192, 4:4096, 9:2048, 18:1024, 36:512}
BRAM_CONF_DEPTH={16384:1, 8192:2, 4096:4, 2048:9, 1024:18, 512:36}

LUTRAM_CONF_WIDTH={16: 32, 8: 64}
LUTRAM_CONF_DEPTH={32: 16, 64: 8}

URAM_CONF_WIDTH={72: 4096}
URAM_CONF_DEPTH={4096: 72}
URAM_EXT_CONF_WIDTH={1:262144, 2:131072, 4:65536, 9:32768, 18:16384, 36:8192, 72:4096}
URAM_EXT_CONF_DEPTH={262144:1, 131072:2, 65536:4, 32768:9, 16384:18, 8192:36, 4096:72}

def bram_array_resource_model(depth, width, array_type, force_bram_pragma=False, detailed=False):
    # based on xilinx forum post: https://forums.xilinx.com/t5/High-Level-Synthesis-HLS/BRAM-usage-large-for-FIFO/m-p/1247118
    # Warning: this estimation only works for ultrascale+ devices, for 7 series devices, the depth will be padded to the powers of 2, https://support.xilinx.com/s/article/61995?language=en_US 

    assert array_type in ['fifo', 'memory']

    # based on vivado synthesis behaviour 
    # hls prediction might differ
    if (depth == 0) or (width == 0) or \
        (array_type == 'fifo' and not force_bram_pragma and width * depth <= 1024) or \
        (array_type == 'memory' and not force_bram_pragma and width * depth < 1024):
        return 0

    # todo: decide resource type in optimiser
    if array_type == 'fifo' and depth <= 32:
        return 0

   # get the number of widths to repeat if greater than max width
    max_width = max(BRAM_CONF_WIDTH.keys()) 
    bram_width = min(max_width, width)

    # find the closest width from the BRAM configuration
    if bram_width not in list(BRAM_CONF_WIDTH.keys()):
        bram_width = sorted(list(BRAM_CONF_WIDTH.keys()))[
                bisect.bisect_right(sorted(list(BRAM_CONF_WIDTH.keys())), bram_width)]

    # get the depth for the bram
    bram_depth = BRAM_CONF_WIDTH[bram_width]

    # return the ceiling
    if detailed:
        return width, bram_width, depth, bram_depth
    else:
        return math.ceil(width/bram_width)*math.ceil(depth/bram_depth)

def uram_array_resource_model(depth, width, extension=True, detailed=False):
    if depth == 0 or width == 0:
        return 0
    if extension:
        max_width = max(URAM_EXT_CONF_WIDTH.keys()) 
        uram_width = min(max_width, width)
        if uram_width not in list(URAM_EXT_CONF_WIDTH.keys()):
            uram_width = sorted(list(URAM_EXT_CONF_WIDTH.keys()))[
                    bisect.bisect_right(sorted(list(URAM_EXT_CONF_WIDTH.keys())), uram_width)]
        uram_depth = URAM_EXT_CONF_WIDTH[uram_width]
    else:
        uram_width = 72
        uram_depth = 4096

    if detailed:
        return width, uram_width, depth, uram_depth
    else:
        return math.ceil(width/uram_width)*math.ceil(depth/uram_depth)


def queue_lutram_resource_model(depth, width):
    if depth == 0 or width == 0:
        return 0
    # find the closest depth from the LUTRAM configuration
    if depth in list(LUTRAM_CONF_DEPTH.keys()):
        lutram_depth = depth
    elif depth > sorted(list(LUTRAM_CONF_DEPTH.keys()))[-1]:
        lutram_depth = sorted(list(LUTRAM_CONF_DEPTH.keys()))[-1]
    else:
        lutram_depth = sorted(list(LUTRAM_CONF_DEPTH.keys()))[
                bisect.bisect_right(sorted(list(LUTRAM_CONF_DEPTH.keys())), depth)]

    # get the depth for the lutram
    lutram_width = LUTRAM_CONF_DEPTH[lutram_depth]

    # return the ceiling
    return math.ceil((width+1)/lutram_width) * math.ceil(depth/lutram_depth)

def dsp_multiplier_resource_model(multiplicand_width, multiplier_width, dsp_type="DSP48E1"):
    #https://github.com/Xilinx/finn/blob/4fee6ffd8e13f91314ec9086e9ce9b2ea9de15c7/src/finn/custom_op/fpgadataflow/streamingfclayer_batch.py#L368,
    # return math.ceil((multiplicand_width+multiplier_width)/48)
    return math.ceil(multiplicand_width/18)*math.ceil(multiplier_width/27)

if __name__ == "__main__":
    print(bram_stream_resource_model(512,4))
    print(bram_stream_resource_model(1024,4))
    print(bram_stream_resource_model(2048,4))
    print(bram_stream_resource_model(4096,4))

    print(bram_stream_resource_model(512,8))
    print(bram_stream_resource_model(1024,8))
    print(bram_stream_resource_model(2048,8))
    print(bram_stream_resource_model(4096,8))

    print(bram_stream_resource_model(512,16))
    print(bram_stream_resource_model(1024,16))
    print(bram_stream_resource_model(2048,16))
    print(bram_stream_resource_model(4096,16))
