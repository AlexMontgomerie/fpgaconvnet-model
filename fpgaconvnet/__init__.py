"""
This repo contains code for optimising the mapping a Convolutional Neural Network (CNN) model to an FPGA. Hardware-specific transforms are applied to the model, producing a hardware description that can be used by a hardware backend, such as [fpgaConvNet HLS](https://github.com/AlexMontgomerie/fpgaconvnet-hls). The generated architecture is streaming-based, and optimised for the specific hardware platform. 

This module is packaged with a command line interface, found in `cli`. This can be used for running optimisation for a given network and platform pair. You can run this using `python -m fpgaconvnet_optimiser`.

An example run would be as follows:

    >>> python -m fpgaconvnet_optimiser --name vgg16 \\
    ...     --model_path examples/models/vgg16.onnx \\
    ...     --platform_path examples/platforms/zc706.json \\
    ...     --output_path outputs/vgg16 \\
    ...     --batch_size 256 \\
    ...     --objective throughput \\
    ...     --transforms fine weights_reloading coarse partition \\
    ...     --optimiser simulated_annealing \\
    ...     --optimiser_config_path examples/optimiser_example.yml 


"""
