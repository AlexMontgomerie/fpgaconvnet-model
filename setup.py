import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fpgaconvnet-model", # Replace with your own username
    version="0.1.4.2",
    author="Alex Montgomerie",
    author_email="am9215@ic.ac.uk",
    description="Parser and model for Convolutional Neural Network Streaming-Based Accelerator on FPGA devices.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexMontgomerie/fpgaconvnet-model",
    include_package_data=True,
    packages=setuptools.find_namespace_packages(
        include=['fpgaconvnet.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "networkx>=2.5",
        "numpy>=1.19.2",
        "protobuf==3.20.3",
        # "torch>=1.11.0",
        # "torchvision>=0.12.0",
        "pyyaml>=5.1.0",
        "scipy>=1.2.1",
        "onnx==1.14.0",
        "onnxruntime>=1.14.1",
        "graphviz>=0.16",
        "pydot>=1.4.2",
        "onnxoptimizer>=0.3.8",
        "ddt>=1.4.2",
        "scikit-learn",
        "coverage==5.5",
        "pyparsing<3"
    ]
)
