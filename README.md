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

This repo can be used to get performance and resource estimates for different hardware configurations. To start, the desired network will need to be parsed into fpgaConvNet's representation. Then a hardware configuration can be loaded, and performance and resource predictions obtained.

```python
from fpgaconvnet.parser import Parser

# initialise network, and load a configuration
parser = Parser(backend="chisel", quant_mode="auto") # use the HLS backend with 16-bit fixed-point quantisation
net = parser.onnx_to_fpgaconvnet("model.onnx") # parse the onnx model

# load existing configuration
net = parser.prototxt_to_fpgaconvnet(net, "config-path.json")

# print performance and resource estimates
print(f"predicted latency (us): {net.get_latency()*1000000}")
print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
print(f"predicted resource usage: {net.partitions[0].get_resource_usage()}")

# visualise the network configuration
net.visualise("image-path.png", mode="png")

# export out the configuration
net.save_all_partitions("config-path.json")
```

## Modelling

In order to do the CNN to hardware mapping, a model of the hardware is needed. There are four levels of abstraction for the final hardware: modules, layers, partitions and network. At each level of abstraction, there is an associated performance and resource estimate so that the constraints for the optimiser can be obtained.

- __Module:__ These are the basic building blocks of the accelerator. The modules are the following:
  - Accum & Accum3D
  - Activation3D
  - AveragePool & AveragePool3D
  - BatchNorm
  - Bias & Bias3D
  - Concat
  - Conv & Conv3D
  - EltWise & EltWise3D
  - Fork & Fork3D
  - Glue & Glue3D
  - MaxPool
  - Pool & Pool3D
  - SlidingWindow & SlidingWindow3D
  - Squeeze & Squeeze3D
  - Stride
  - VectorDot & VectorDot3D
- __Layer:__ Layers are comprised of modules. They have the same functionality of the equivalent layers of the CNN model. The following layers are supported:
  - Activation 3D
  - AveragePooling & AveragePooling 3D
  - Batch Normalization
  - Concatenation
  - Convolution & Convolution 3D
  - Element Wise & Element Wise 3D
  - Inner Product & Inner Product 3D
  - Pooling & Pooling 3D
  - ReLU & ReLU 3D
  - Split & Split 3D
  - Squeeze & Squeeze 3D
- __Partition:__ Partitions make up a sub-graph of the CNN model network. They are comprised of layers. A single partition fits on an FPGA at a time, and partitions are changed by reconfiguring the FPGA.
- __Network:__ This is the entire CNN model described through hardware. A network contains partitions and information on how to execute them.

---

Feel free to post an issue if you have any questions or problems!
