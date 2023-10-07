import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

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
    python_requires='>=3.9',
    install_requires=required_packages,

)
