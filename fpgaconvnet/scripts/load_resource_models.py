import os
import importlib
import inspect

import numpy as np
import matplotlib.pyplot as plt

from fpgaconvnet.tools.resource_regression_model import ModuleModel

CHISEL_MODULES = {
        "Accum": "AccumFixed",
        # "Fork": "ForkFixed",
        "Glue": "GlueFixed",
        "SlidingWindow": "SlidingWindowFixed",
        "SlidingWindow3D": "SlidingTensorFixed",
        "Squeeze": "SqueezeFixed",
        "VectorDot": "VectorDotFixed",
        "Pool": "MaxPoolFixed",
        "GlobalPool": "AveragePoolFixed",
        "Bias": "BiasFixed"
}

# CHISEL_MODULES = [ "AveragePool" ]
HLS_MODULES = []

# iterate over chisel modules
for module, identifier in CHISEL_MODULES.items():

    print(f"{module} (chisel)")
    # create regression model
    rsc_model = ModuleModel(identifier, module, "chisel")

    # load data
    rsc_model.load_data_from_db()

    # fit model
    rsc_model.fit_model(from_cache=False)

    # save coefficeints
    cache_path = os.path.join(
            os.path.dirname(__file__),
            f"../coefficients/chisel")
    rsc_model.save_coef(cache_path)

    # get modelled resources
    hw_model = getattr(importlib.import_module(
        f"fpgaconvnet.models.modules.{module}"), module)

    # set defaults for module
    attrs = inspect.getmembers(hw_model, lambda x:not(inspect.isroutine(x)))
    attrs = [x for x in attrs if (x[0] == "__dataclass_fields__")]
    attrs = {k : 1 for k, v in attrs[0][1].items() if v.init}
    attrs["backend"] = "chisel"
    attrs.pop("pool_type", None) # bug fix for Pool
    m = hw_model(**attrs)

    # get actual resource cost
    predicted = { "LUT": [], "Logic_LUT": [], "LUT_RAM": [],
            "LUT_SR": [], "FF": [], "BRAM": [], "DSP": [], }
    actual = { "LUT": [], "Logic_LUT": [], "LUT_RAM": [],
            "LUT_SR": [], "FF": [], "BRAM": [], "DSP": [], }
    for index, param in enumerate(rsc_model.parameters):

        # set specific parameters
        for k, v in param.items():
            if hasattr(m, k):
                setattr(m, k, v)

        # get the predicted resources
        rsc = m.rsc(coef=rsc_model.coef)
        for rsc_type in predicted.keys():
            predicted[rsc_type].append(rsc[rsc_type])

        # get the actual resources
        actual["Logic_LUT"].append(rsc_model.actual["Logic_LUT"][index])
        actual["LUT_RAM"].append(rsc_model.actual["LUT_RAM"][index])
        actual["LUT_SR"].append(rsc_model.actual["LUT_SR"][index])
        actual["LUT"].append(
            rsc_model.actual["Logic_LUT"][index] +
            rsc_model.actual["LUT_RAM"][index] +
            rsc_model.actual["LUT_SR"][index]
        )
        actual["FF"].append(rsc_model.actual["FF"][index])
        actual["DSP"].append(rsc_model.actual["DSP"][index])
        actual["BRAM"].append(
            2*rsc_model.actual["BRAM36"][index] +
            rsc_model.actual["BRAM18"][index]
        )

    for rsc_type in predicted.keys():
        diff = np.array(predicted[rsc_type]) - np.array(actual[rsc_type])
        mean_err = diff.mean()
        mse = (diff*diff).mean()
        std = diff.std()

        print(f"({module}) {rsc_type}: mean_err={mean_err:.2f}, mse={mse:.2f}, std={std:.2f}")

        x = actual[rsc_type]
        y = predicted[rsc_type]
        plt.plot(x, x, label="actual")
        plt.scatter(x, y, label="predict", marker="x", color="r")
        plt.title(rsc_type)
        plt.xlabel("actual")
        plt.ylabel("predicted")
        plt.legend()
        # plt.show()

#     for rsc_type in self.rsc_types:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         x = self.actual[rsc_type]
    #         y = self.predict[rsc_type]
    #         ax.plot(x, x, label="actual")
    #         ax.scatter(x, y, label="predict", marker="x", color="r")
    #         ax.set_title(rsc_type)
    #         ax.set_xlabel("actual")
    #         ax.set_ylabel("predicted")
    #         ax.legend()

    #         filepath = os.path.join(outpath, f"{self.name}_{rsc_type}.jpg".lower())
    #         fig.savefig(filepath)


