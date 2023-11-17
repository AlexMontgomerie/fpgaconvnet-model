from abc import ABC, abstractmethod
import pickle

import numpy as np
import sklearn.metrics

from fpgaconvnet.models.modules import ModuleBase
from fpgaconvnet.models.modules.database import Record

class ResourceFitter(ABC):

    @abstractmethod
    def fit(self, data: Record):
        pass

    @abstractmethod
    def eval(self, module: ModuleBase):
        pass

    def save_fitter(self):
        return pickle.dumps(self.fitter)

    def load_fitter(self, coef: bytes):
        return pickle.loads(coef)

    def get_accuracy(self, train: Record, test: Record, rsc_type: str):

        # fit to the training data
        self.fit(train, rsc_type)

        # get the predictions and golden data
        test_predictions = [self.eval(m, rsc_type) for m in test.modules() ]
        test_golden = test.resources(rsc_type)

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


class NNLSResourceFitter(ResourceFitter):

    import sklearn.linear_model

    def __init__(self):

        # instance of the NNLS fitter
        self.fitter = self.sklearn.linear_model.LinearRegression(
                positive=True, fit_intercept=False)

        # dictionary to store coefficients
        self.coefficients = {}

    def fit(self, data: Record, rsc_type: str):
        parameters = data.parameters()
        resources = data.resources(rsc_type)
        self.fitter.fit(parameters, resources)
        self.coefficients[rsc_type] = self.save_fitter()

    def eval(self, module: ModuleBase, rsc_type: str):

        # try to get the coefficients
        try:
            coefficients = self.coefficients[rsc_type]
        except KeyError:
            raise ValueError("Resource type not fitted")

        # update the fitter with coefficients
        self.fitter = self.load_fitter(coefficients)

        # return the prediction
        return self.fitter.predict([module.resource_parameters()])[0]


class NNLSHeuristicResourceFitter(ResourceFitter):

    import sklearn.linear_model

    def __init__(self):

        # instance of the NNLS fitter
        self.fitter = self.sklearn.linear_model.LinearRegression(
                positive=True, fit_intercept=False)

        # dictionary to store coefficients
        self.coefficients = {}

    def fit(self, data: Record, rsc_type: str):
        parameters = data.heuristic_parameters(rsc_type)
        resources = data.resources(rsc_type)
        self.fitter.fit(parameters, resources)
        self.coefficients[rsc_type] = self.save_fitter()

    def eval(self, module: ModuleBase, rsc_type: str):

        # try to get the coefficients
        try:
            coefficients = self.coefficients[rsc_type]
        except KeyError:
            raise ValueError("Resource type not fitted")

        # update the fitter with coefficients
        self.fitter = self.load_fitter(coefficients)

        # get the resource parameters
        rsc_params = module.resource_parameters_heuristics()[rsc_type]

        # return the prediction
        return self.fitter.predict([rsc_params])[0]

class SVRResourceFitter(ResourceFitter):

    import sklearn.svm

    def __init__(self, kernel="rbf", degree=3):

        # instance of the NNLS fitter
        self.fitter = self.sklearn.svm.SVR(
                kernel=kernel, degree=degree, gamma="scale")

        # dictionary to store coefficients
        self.coefficients = {}

    def fit(self, data: Record, rsc_type: str):
        parameters = data.parameters()
        resources = data.resources(rsc_type)
        self.fitter.fit(parameters, resources)
        self.coefficients[rsc_type] = self.save_fitter()

    def eval(self, module: ModuleBase, rsc_type: str):

        # try to get the coefficients
        try:
            coefficients = self.coefficients[rsc_type]
        except KeyError:
            raise ValueError("Resource type not fitted")

        # update the fitter with coefficients
        self.fitter = self.load_fitter(coefficients)

        # return the prediction
        return self.fitter.predict([module.resource_parameters()])[0]


