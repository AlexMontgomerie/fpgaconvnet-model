# fpgaConvNet Model

This repo contains performance and resource for the building blocks of fpgaConvNet, a Streaming Architecture-based Convolutional Neural Network (CNN) acceleration toolflow, which maps CNN models to FPGAs. The building blocks are implemented in hardware in the [fpgaconvnet-hls](https://github.com/AlexMontgomerie/fpgaconvnet-hls) repository. These models are used in conjunction with [samo](https://github.com/AlexMontgomerie/samo), a Streaming Architecture optimiser, where there are instructions for performing optimisation.

## Setup

The following programs are required:

- `python (>=3.7)`

To install this package, run from this directory the following:

```
python -m pip install fpgaconvnet-model
```

## Usage

The 

## Modelling

In order to do the CNN to hardware mapping, a model of the hardware is needed. There are four levels of abstraction for the final hardware: modules, layers, partitions and network. At each level of abstraction, there is an associated performance and resource estimate so that the constraints for the optimiser can be obtained.

- __Module:__ These are the basic building blocks of the accelerator. The modules are the following:
  - Accum
  - BatchNorm
  - Conv
  - Glue
  - SlidingWindow
  - Fork
  - Pool
  - Squeeze
- __Layer:__ Layers are comprised of modules. They have the same functionality of the equivalent layers of the CNN model. The following layers are supported:
  - Batch Normalization
  - Convolution
  - Inner Product
  - Pooling
  - ReLU
- __Partition:__ Partitions make up a sub-graph of the CNN model network. They are comprised of layers. A single partition fits on an FPGA at a time, and partitions are changed by reconfiguring the FPGA.
- __Network:__ This is the entire CNN model described through hardware. A network contains partitions and information on how to execute them.

---

Feel free to post an issue if you have any questions or problems!
