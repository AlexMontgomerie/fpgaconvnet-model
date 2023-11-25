#!/bin/bash

MODULES=( "accum" "relu" "squeeze" "sliding_window" "pad" "pool" "hardswish" "concat" "fork" "glue" "vector_dot" "bias" )
# MODULES=( "bias" )

# iterate over the modules
for m in ${MODULES[@]}; do

    # regenerate resource models
    python -m fpgaconvnet.models.modules -n default -r LUT  -m $m -b chisel -d 2 -c scripts/default_model.toml
    python -m fpgaconvnet.models.modules -n default -r FF   -m $m -b chisel -d 2 -c scripts/default_model.toml
    python -m fpgaconvnet.models.modules -n default -r BRAM -m $m -b chisel -d 2 -c scripts/default_model.toml
    # python -m fpgaconvnet.models.modules -n default -r DSP  -m $m -b chisel -d 2 -c scripts/default_model.toml

done
