# fpgaConvNet Optimiser

This repo contains code for optimising the mapping a Convolutional Neural Network (CNN) model to an FPGA. Hardware-specific transforms are applied to the model, producing a hardware description that can be used by a hardware backend, such as [fpgaConvNet HLS](https://github.com/AlexMontgomerie/fpgaconvnet-hls). The generated architecture is streaming-based, and optimised for the specific hardware platform.

## Setup

The following programs are required:

- `python=3.7`

To install this package, run from this directory the following:

```
sudo apt-get install protobuf-compiler libprotoc-dev
python -m pip install .
```

## Testing

A suite of tests have been created for the optimiser repo. To run all of them, use the following:

```
python -m unittest discover tests/
```

## Optimiser Framework

The main tool is the optimisation script which generates an optimised hardware topology for a given model and platform. There are several components needed for this: a model of the hardware, transforms that map the model to the hardware and an optimisation scheme that chooses the best mapping. These will be outlined later.
To use the optimiser, an example of running it using the `run_optimiser.py` script for VGG16 is as follows:

```Shell
python -m fpgaconvnet_optimiser --name vgg16 \
    --model_path examples/models/vgg16.onnx \
    --platform_path examples/platforms/zc706.json \
    --output_path outputs/vgg16 \
    --batch_size 256 \
    --objective throughput \
    --optimiser simulated_annealing \
    --optimiser_config_path examples/optimiser_example.yml
```

This will generate the following files:

- `(output_path)/(name).prototxt`: Hardware topology description for backend
- `(output_path)/report.json`: A report file containing estimations of usage and performance
- `(output_path)/scheduler.csv`: A schedule for running partitions as well as information for memory management
- `(output_path)/topology.png`: Visualisation of the hardware topology

These files in the output directory can be used with [fpgaConvNet HLS](https://github.com/AlexMontgomerie/fpgaconvnet-hls) to generate the actual hardware and run on the board.

### Modelling

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

### Transforms

In order to find the optimal mapping, transforms are used to manipulate the performance/resource trade-off of the hardware. The main transforms implemented currently are:

| Transform | Level | Description |
|-----------|:-----:|------------|
| Fine | Module | Defines the parallelism for the kernel x kernel dot product of the `conv` module. |
| Coarse | Layer | Input and output channel dimension parallelism of Layers. For a convolution layer, this is how filters are run in parallel. |
| Weights Reloading | Partition | Reduces on-chip memory usage by creating partial featuremaps across partition iterations. |
| Partitioning | Network | Defines how the graph is split into subgraphs of the model for different reconfigurable components. |

### Optimisation Schemes

Finally, optimisations schemes are used to explore the transform design space and find an optimal mapping for a given hardware platform. The optimisation schemes implemented are the following:

- __Simulated Annealing:__ Randomly chooses a transform and hardware component to change. The change is accepted based on a probability-based decision function.
- __Improve:__ Chooses the hardware component causing a bottleneck and performs the same decision as simulated annealing.

## Citations

If you use this work, please use the following references:

```BibTex
@article{venieris_fpgaconvnet_2019,
    title = {fpgaConvNet: Mapping Regular and Irregular Convolutional Neural Networks on FPGAs},
    journal = {IEEE Transactions on Neural Networks and Learning Systems},
    author = {Venieris, S. I. and Bouganis, C.},
    year = {2019},
}

@inproceedings{venieris_fpgaconvnet_2017,
    title = {fpgaConvNet: A Toolflow for Mapping Diverse Convolutional Neural Networks on Embedded FPGAs},
    booktitle = {NIPS 2017 Workshop on Machine Learning on the Phone and other Consumer Devices},
    author = {Venieris, Stylianos I. and Bouganis, Christos-Savvas},
    year = {2017},
}

@inproceedings{venieris_fpgaconvnet_2016,
    title = {fpgaConvNet: A Framework for Mapping Convolutional Neural Networks on FPGAs},
    booktitle = {2016 IEEE 24th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)},
    author = {Venieris, S. I. and Bouganis, C.},
    year = {2016},
}
```

---

Feel free to post an issue if you have any questions or problems!
