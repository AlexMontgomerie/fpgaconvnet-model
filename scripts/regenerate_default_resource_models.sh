#!/bin/bash

MODULES=( "accum" )

# iterate over the modules
for m in ${MODULES[@]}; do

    # regenerate resource models
    python -m fpgaconvnet.models.modules -n default -r LUT  -m $m -b chisel -d 2 -c test_rsc.toml
    python -m fpgaconvnet.models.modules -n default -r FF   -m $m -b chisel -d 2 -c test_rsc.toml
    python -m fpgaconvnet.models.modules -n default -r BRAM -m $m -b chisel -d 2 -c test_rsc.toml
    python -m fpgaconvnet.models.modules -n default -r DSP  -m $m -b chisel -d 2 -c test_rsc.toml

done
