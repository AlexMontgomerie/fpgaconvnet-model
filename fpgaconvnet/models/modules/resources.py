from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import os
from typing import Optional
from functools import singledispatch

import numpy as np
import sklearn.metrics

from fpgaconvnet.models.modules import ModuleBaseMeta, ModuleBase, CHISEL_RSC_TYPES
from fpgaconvnet.models.modules.database import Record

@dataclass(kw_only=True)
class ResourceModel(ABC):
    module_type: ModuleBaseMeta
    rsc_type: str
    cache_path: str = ""

    def __post_init__(self):
        if self.cache_path:
            self.model = self.load_model(self.cache_path)

    @abstractmethod
    def fit(self, data: Record):
        pass

    @abstractmethod
    def eval(self, module: ModuleBase) -> int:
        pass

    def __call__(self, module: ModuleBase) -> int:
        return self.eval(module)

    def save_model(self, cache_path: str):
        with open(cache_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(cache_path: str):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def get_accuracy(self, train: Record, test: Record):

        # fit to the training data
        self.fit(train)

        # get the predictions and golden data
        test_predictions = [self.eval(m) for m in test.modules() ]
        test_golden = test.resources(self.rsc_type)

        # collect all the accuracy metrics
        mae = sklearn.metrics.mean_absolute_error(test_golden, test_predictions)
        mse = sklearn.metrics.mean_squared_error(test_golden, test_predictions)
        r2 = sklearn.metrics.r2_score(test_golden, test_predictions)

        # return the accuracy metrics
        return {
            "mae": mae,
            "mse": mse,
            "r2": r2
        }


class NNLSResourceModel(ResourceModel):

    import sklearn.linear_model

    def __post_init__(self):

        # instance of the NNLS model
        self.model = self.sklearn.linear_model.LinearRegression(
                positive=True, fit_intercept=False)

        # call the parent post init
        super().__post_init__()

    def fit(self, data: Record):
        parameters = data.parameters()
        resources = data.resources(self.rsc_type)
        self.model.fit(parameters, resources)

    def eval(self, module: ModuleBase) -> int:
        return self.model.predict([module.resource_parameters()])[0]


class NNLSHeuristicResourceModel(ResourceModel):

    import sklearn.linear_model

    def __post_init__(self):

        # instance of the NNLS model
        self.model = self.sklearn.linear_model.LinearRegression(
                positive=True, fit_intercept=False)

        # call the parent post init
        super().__post_init__()

    def fit(self, data: Record):
        parameters = data.heuristic_parameters(self.rsc_type)
        resources = data.resources(self.rsc_type)
        self.model.fit(parameters, resources)

    def eval(self, module: ModuleBase):

        # get the resource parameters
        rsc_params = module.resource_parameters_heuristics()[self.rsc_type]

        # return the prediction
        return self.model.predict([rsc_params])[0]


class SVRResourceModel(ResourceModel):
    kernel: str = "rbf"
    degree: int = 3

    import sklearn.svm

    def __post_init__(self):

        # instance of the NNLS model
        self.model = self.sklearn.svm.SVR(
                kernel=self.kernel,
                degree=self.degree,
                gamma="scale")

        # call the parent post init
        super().__post_init__()

    def fit(self, data: Record):
        parameters = data.parameters()
        resources = data.resources(self.rsc_type)
        self.model.fit(parameters, resources)

    def eval(self, module: ModuleBase):
        # return the prediction
        return self.model.predict([module.resource_parameters()])[0]


@singledispatch
def eval_resource_model(m: ModuleBase, rsc_type: str, model: Optional[ResourceModel] = None) -> int:
    assert rsc_type == model.rsc_type, f"Incompatible resource type with model: {rsc_type}"
    return model(m)

def get_cached_resource_model(m: ModuleBaseMeta, rsc_type: str, name: str) -> dict[str, ResourceModel]:

    # get the cache path
    # filename = f"{name}.{rsc_type}.{m.name}.{m.backend.name}.{list(m.dimensionality)[0].value}.model"
    filename = f"{name}.{rsc_type}.{m.name}.{m.backend.name.lower()}.2.model"
    cache_path = os.path.join(os.path.dirname(__file__), "..", "cache", "modules", filename)
    model = ResourceModel.load_model(cache_path)

    # return the models
    return model


# ResourceEvaluate: Callable[[ModuleBase], dict]
# Fit: Callable[[Record], ResourceEvaluate]


# def fit_resource_model(

# class MyModel:
#     def __init__():...
#     def _some_internal_calc: ...
#     def __call__(self) -> ResourceEvaluator:

# mymodel = MyModel()

# @singledispatch
# def eval_resources(m: ModuleBase, model: ResourceModel) -> dict:
#     ...

# def imp_a(m) -> dict:
#     ....


# @eval_resources.register
# def _(m: AccumChisel | ForkChisel, ) -> dict:
#     # Your implementation here
#     # Implementation A
#     # impa_a(m)

# @eval_resources.register
# def _(m: ForkChisel) -> dict:
#     # Your implementation here
#     # Implementation A
#     # impa_a(m)


# eval_resources(module)



