#!/bin/bash

CHISEL_MODULES=( "accum" "relu" "squeeze" "sliding_window" "pad" "pool" "hardswish" "concat" "fork" "glue" "vector_dot" "bias" )
HLS_MODULES=( "accum" "relu" "squeeze" "sliding_window" "pool" "fork" "glue" "conv" "bias" )
# MODULES=( "accum" )

for m in ${CHISEL_MODULES[@]}; do

    # regenerate resource models
    python -m fpgaconvnet.models.modules -n default -m $m -b chisel -d 2 -c scripts/default_chisel_model.toml -p fpgaconvnet/platform/configs/zcu104.toml

done

for m in ${HLS_MODULES[@]}; do

    # regenerate resource models
    python -m fpgaconvnet.models.modules -n default -m $m -b hls -d 2 -c scripts/default_hls_model.toml -p fpgaconvnet/platform/configs/zc706.toml
    # python -m fpgaconvnet.models.modules -n default -m $m -b hls -d 3 -c scripts/default_hls_model.toml -p fpgaconvnet/platform/configs/zc706.toml

done
