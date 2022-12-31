import os
import importlib
import inspect

import numpy as np
import matplotlib.pyplot as plt

from fpgaconvnet.tools.resource_regression_model import ModuleModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, mean_absolute_percentage_error

# Available regression models: linear_regression, xgboost
REGRESSOR = "linear_regression"

# CHISEL_MODULES = {
#         "Accum": "AccumFixed",
#         "Fork": "ForkFixed",
#         "Glue": "GlueFixed",
#         # "SlidingWindow": "SlidingWindowFixed",
#         "SlidingWindow3D": "SlidingTensorFixed",
#         "Squeeze": "SqueezeFixed",
#         "VectorDot": "VectorDotFixed",
#         "Pool": "MaxPoolFixed",
#         # "GlobalPool": "AveragePoolFixed",
#         # "Bias": "BiasFixed"
# }

CHISEL_MODULES = {
        "Accum": "Accum",
        "Fork": "Fork",
        "Glue": "Glue",
        # "SlidingWindow": "SlidingWindowFixed",
        "SlidingWindow3D": "SlidingTensor",
        "Squeeze": "Squeeze",
        "VectorDot": "VectorDot",
        "Pool": "MaxPool",
        # "GlobalPool": "AveragePoolFixed",
        # "Bias": "BiasFixed"
}

# CHISEL_MODULES = [ "AveragePool" ]
HLS_MODULES = []

def save_npy(rsc_model, module):
    db_data_local_path = os.path.join(os.path.dirname(__file__), f"../db_data")
    if not os.path.exists(db_data_local_path):
        os.mkdir(db_data_local_path)
    X = []
    Y = []
    for param, actual in zip(rsc_model.parameters, rsc_model.actual["Logic_LUT"]):
        x = []
        for k, p in param.items():
            if isinstance(p, list):
                x.extend(p)
            elif isinstance(p, str):
                continue
            else:
                x.append(p)
        X.append(x)
        Y.append([actual])

    x_arr = np.array(X)
    y_arr = np.array(Y)
    np.save(f"{db_data_local_path}/{module}_Logic_LUT_X.npy", x_arr)
    np.save(f"{db_data_local_path}/{module}_Logic_LUT_Y.npy", y_arr)

    X = []
    Y = []
    for param, actual in zip(rsc_model.parameters, rsc_model.actual["LUT_RAM"]):
        x = []
        for k, p in param.items():
            if isinstance(p, list):
                x.extend(p)
            elif isinstance(p, str):
                continue
            else:
                x.append(p)
        X.append(x)
        Y.append([actual])

    x_arr = np.array(X)
    y_arr = np.array(Y)
    np.save(f"{db_data_local_path}/{module}_LUT_RAM_X.npy", x_arr)
    np.save(f"{db_data_local_path}/{module}_LUT_RAM_Y.npy", y_arr)

    X = []
    Y = []
    for param, actual in zip(rsc_model.parameters, rsc_model.actual["LUT_SR"]):
        x = []
        for k, p in param.items():
            if isinstance(p, list):
                x.extend(p)
            elif isinstance(p, str):
                continue
            else:
                x.append(p)
        X.append(x)
        Y.append([actual])

    x_arr = np.array(X)
    y_arr = np.array(Y)
    np.save(f"{db_data_local_path}/{module}_LUT_SR_X.npy", x_arr)
    np.save(f"{db_data_local_path}/{module}_LUT_SR_Y.npy", y_arr)

    X = []
    Y = []
    for param, actual in zip(rsc_model.parameters, rsc_model.actual["FF"]):
        x = []
        for k, p in param.items():
            if isinstance(p, list):
                x.extend(p)
            elif isinstance(p, str):
                continue
            else:
                x.append(p)
        X.append(x)
        Y.append([actual])

    x_arr = np.array(X)
    y_arr = np.array(Y)
    np.save(f"{db_data_local_path}/{module}_FF_X.npy", x_arr)
    np.save(f"{db_data_local_path}/{module}_FF_Y.npy", y_arr)

    X = []
    Y = []
    for param, actual in zip(rsc_model.parameters, rsc_model.actual["DSP"]):
        x = []
        for k, p in param.items():
            if isinstance(p, list):
                x.extend(p)
            elif isinstance(p, str):
                continue
            else:
                x.append(p)
        X.append(x)
        Y.append([actual])

    x_arr = np.array(X)
    y_arr = np.array(Y)
    np.save(f"{db_data_local_path}/{module}_DSP_X.npy", x_arr)
    np.save(f"{db_data_local_path}/{module}_DSP_Y.npy", y_arr)

    X = []
    Y = []
    for param, actual in zip(rsc_model.parameters, rsc_model.actual["BRAM36"]):
        x = []
        for k, p in param.items():
            if isinstance(p, list):
                x.extend(p)
            elif isinstance(p, str):
                continue
            else:
                x.append(p)
        X.append(x)
        Y.append([actual])

    x_arr = np.array(X)
    y_arr = np.array(Y)
    np.save(f"{db_data_local_path}/{module}_BRAM36_X.npy", x_arr)
    np.save(f"{db_data_local_path}/{module}_BRAM36_Y.npy", y_arr)

    X = []
    Y = []
    for param, actual in zip(rsc_model.parameters, rsc_model.actual["BRAM18"]):
        x = []
        for k, p in param.items():
            if isinstance(p, list):
                x.extend(p)
            elif isinstance(p, str):
                continue
            else:
                x.append(p)
        X.append(x)
        Y.append([actual])

    x_arr = np.array(X)
    y_arr = np.array(Y)
    np.save(f"{db_data_local_path}/{module}_BRAM18_X.npy", x_arr)
    np.save(f"{db_data_local_path}/{module}_BRAM18_Y.npy", y_arr)

# iterate over chisel modules
for module, identifier in CHISEL_MODULES.items():

    print(f"{module} (chisel) ({REGRESSOR})")
    # create regression model
    rsc_model = ModuleModel(identifier, module, REGRESSOR, "chisel")

    # load data
    rsc_model.load_data_from_db()

    # save database data locally
    # save_npy(rsc_model, module)

    # fit model
    rsc_model.fit_model(from_cache=False)

    # save coefficeints
    cache_path = os.path.join(
            os.path.dirname(__file__),
            f"../coefficients/{REGRESSOR}/chisel")
    rsc_model.save_coef(cache_path)

    # get modelled resources
    hw_model = getattr(importlib.import_module(
        f"fpgaconvnet.models.modules.{module}"), module)

    # set defaults for module
    attrs = inspect.getmembers(hw_model, lambda x:not(inspect.isroutine(x)))
    attrs = [x for x in attrs if (x[0] == "__dataclass_fields__")]
    attrs = {k : 1 for k, v in attrs[0][1].items() if v.init}
    attrs["backend"] = "chisel"
    attrs["regression_model"] = REGRESSOR
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
        match REGRESSOR:
            case "linear_regression":
                rsc = m.rsc(coef=rsc_model.coef)
            case "xgboost" | "xgboost-kernel":
                rsc = m.rsc(model=rsc_model.coef)

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

        mse = mean_squared_error(actual[rsc_type], predicted[rsc_type])
        mae = mean_absolute_error(actual[rsc_type], predicted[rsc_type])
        mape = mean_absolute_percentage_error(actual[rsc_type], predicted[rsc_type])
        evs = explained_variance_score(actual[rsc_type], predicted[rsc_type])
        r2 = r2_score(actual[rsc_type], predicted[rsc_type])

        print(f"({module}) {rsc_type}: mse={mse:.2f}, mae={mae:.2f}, mape={mape:.2f}, evs={evs:.2f}, r2={r2:.2f}")

        x = actual[rsc_type]
        y = predicted[rsc_type]
        plt.clf()
        plt.plot(x, x, label="actual")
        plt.scatter(x, y, label="predict", marker="x", color="r")
        plt.title(rsc_type)
        plt.xlabel("actual")
        plt.ylabel("predicted")
        plt.legend()
        # imgs_path = os.path.join(os.path.dirname(__file__), f"../imgs")
        # plt.savefig(os.path.join(imgs_path, f"{REGRESSOR}_{module}_{rsc_type}.jpg"))
        # plt.show()

    # for rsc_type in self.rsc_types:
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


