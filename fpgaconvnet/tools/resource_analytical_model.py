import bisect
import math

BRAM_CONF_WIDTH={1:16384, 2:8192, 4:4096, 9:2048, 18:1024, 36:512}
BRAM_CONF_DEPTH={16384:1, 8192:2, 4096:4, 2048:9, 1024:18, 512:36}

BRAM36_CONF_WIDTH={1:32000, 2:16000, 4:8000, 9:4000, 18:2000, 36:1000}
BRAM36_CONF_DEPTH={32000:1, 16000:2, 8000:9, 4000:9, 2000:18, 1000:36}

LUTRAM_CONF_WIDTH={16: 32, 8: 64}
LUTRAM_CONF_DEPTH={32: 16, 64: 8}

def bram_efficient_resource_model(depth, width):
    # based on xilinx forum post: https://forums.xilinx.com/t5/High-Level-Synthesis-HLS/BRAM-usage-large-for-FIFO/m-p/1247118

    assert width > 0, "width must be greater than zero"
    assert width <= 36, "width must be less than 36"

    if math.ceil(math.log(depth, 2)) < 5:
        return 0

    # if there is zero depth, return no BRAM usage
    if depth == 0:
        return 0

    # find the closest depth from the BRAM configuration
    bram_depth = min(BRAM_CONF_DEPTH.keys(),
            key=lambda x: abs(x-depth))

    # get the depth for the bram
    bram_width = BRAM_CONF_DEPTH[bram_depth]

    # bram usage
    bram = math.ceil(width/bram_width) * math.ceil(depth/bram_depth)

    if depth - bram_depth > 0:
        return bram + bram_efficient_resource_model(depth - bram_depth, width)
    else:
        return bram

    # # return the ceiling
    # return math.ceil(width/bram_width) * math.ceil(depth/bram_depth)



def bram_stream_resource_model(depth, width):
    # based on xilinx forum post: https://forums.xilinx.com/t5/High-Level-Synthesis-HLS/BRAM-usage-large-for-FIFO/m-p/1247118

    assert width > 0, "width must be greater than zero"
    assert width <= 36, "width must be less than 36"

    if math.ceil(math.log(depth, 2)) < 5:
        return 0

    # if there is zero depth, return no BRAM usage
    if depth == 0:
        return 0

    # find the closest depth from the BRAM configuration
    if depth in list(BRAM_CONF_DEPTH.keys()):
        bram_depth = depth
    elif depth > sorted(list(BRAM_CONF_DEPTH.keys()))[-1]:
        bram_depth = sorted(list(BRAM_CONF_DEPTH.keys()))[-1]
    else:
        bram_depth = sorted(list(BRAM_CONF_DEPTH.keys()))[
                bisect.bisect_right(sorted(list(BRAM_CONF_DEPTH.keys())), depth)]

    # get the depth for the bram
    bram_width = BRAM_CONF_DEPTH[bram_depth]

    # return the ceiling
    return math.ceil(width/bram_width) * math.ceil(depth/bram_depth)

def bram_memory_resource_model(depth, width):
    assert width > 0, "width must be greater than zero"
    # assert width <= 36, "width must be less than 36"

    # if math.ceil(math.log(depth, 2)) <= 5:
    #     return 0

    # get the number of widths to repeat if greater than max width
    max_width = max(BRAM_CONF_WIDTH.keys())
    repeated_bram = math.ceil(width/max_width)
    width = min(max_width, width)

    # if there is zero depth, return no BRAM usage
    if depth == 0:
        return 0

    # find the closest width from the BRAM configuration
    if width in list(BRAM_CONF_WIDTH.keys()):
        bram_width = width
    else:
        bram_width = sorted(list(BRAM_CONF_WIDTH.keys()))[
                bisect.bisect_right(sorted(list(BRAM_CONF_WIDTH.keys())), width)]

    # get the depth for the bram
    bram_depth = BRAM_CONF_WIDTH[bram_width]

    # return the ceiling
    return repeated_bram*math.ceil(depth/bram_depth)

def queue_lutram_resource_model(depth, width):

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
