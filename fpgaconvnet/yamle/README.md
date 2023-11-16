

## Setup
Clone and pip install the "fpgaconvnet-model" [dev-yamle](https://github.com/AlexMontgomerie/fpgaconvnet-model/tree/dev-yamle) branch and "fpgaconvnet-optimiser" at [this_commit](https://github.com/AlexMontgomerie/fpgaconvnet-optimiser/tree/65c6687b4ba6672235e32c7e2d7a148381732237).


## Example
Example path
```
./
├─ fpgaconvnet-model/
├─ fpgaconvnet-optimiser/
│  ├─ examples/
│  │  ├─ u250.toml
├─ models/
|  ├─ model_0.onnx
├─ example.py


```
example.py

```python
from fpgaconvnet.yamle.yamle_interface import get_yamle_model_cost

platform_path = "./fpgaconvnet-optimiser/examples/platforms/u250.toml"                                      
model_path = "./models/model_0.onnx"

get_yamle_model_cost(model_path, platform_path)

```