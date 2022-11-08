import os
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
import importlib
import inspect
import dataclasses
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from fpgaconvnet.models.modules import Module

RSC_TYPES=["LUT"]

SERVER_DB="mongodb+srv://fpgaconvnet.hwnxpyo.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"

class ModuleModel:
    def __init__(self, name, backend, filepath=".pem"):
        self.name = name
        self.backend = backend

        self.parameters = []
        self.model = {k: [] for k in RSC_TYPES}
        self.coef = {k: [] for k in RSC_TYPES}
        self.predict = {k: [] for k in RSC_TYPES}
        self.actual = {k: [] for k in RSC_TYPES}


        if filepath.endswith(".pem"):
            self.load_data_from_db(filepath)
        else:
            self.load_data_from_dir(filepath)

    def load_data_from_db(self, filepath):
        """
        loads a set of parameter-resource pairs for
        many different configurations and runs of the
        module. This is to be used with the
        `fit_resource_model` method.
        """
        # database .pem path
        db_pem = os.path.join(os.path.dirname(__file__),
                "fpgaconvnet-mongodb.pem")

        # create MongoDB client
        client = MongoClient(SERVER_DB, tls=True,
                tlsCertificateKeyFile=db_pem,
                server_api=ServerApi('1'))

        # load database collection for given backend
        db = client["fpgaconvnet"]
        collection = db[self.backend]

        filter = {"name":self.name}
        if self.backend == "chisel":
            filter = {"name":self.name+"Fixed"}

        for document in collection.find(filter):
            self.parameters.append(document["parameters"])
            for rsc in RSC_TYPES:
                self.actual[rsc].append(document["resources"][rsc])

        assert len(self.parameters) > 0

    def load_data_from_dir(self, filepath):
        folders = [f for f in os.listdir(filepath) if not f]
        for folder in folders:
            filename = os.path.join(filepath, folder, "results.json")
            try:
                with open(filename,"r") as jsonfile:
                    oneDict = json.load(jsonfile)
                    self.parameters.append(oneDict["parameters"])
                    for rsc in RSC_TYPES:
                        self.actual[rsc].append(oneDict["resources"][rsc])
            except:
                pass

        assert len(self.parameters) > 0

    def fit_model(self):
        module = getattr(importlib.import_module(
            f"fpgaconvnet.models.modules.{self.name}"), self.name)

        # hack: set default for missing fields
        attrs = inspect.getmembers(module, lambda x:not(inspect.isroutine(x)))
        attrs = [x for x in attrs if (x[0] == "__dataclass_fields__")]
        attrs = {k : 1 for k, v in attrs[0][1].items() if v.init}
        m = module(**attrs)

        for point in self.parameters:
            for k, v in point.items():
                if hasattr(m, k):
                    setattr(m,k,v)
                # hack: ignore fields if module doesn't care

            for rsc_type in RSC_TYPES:
                self.model[rsc_type].append(m.utilisation_model()[rsc_type])

        for rsc_type in RSC_TYPES:
            self.coef[rsc_type] = self.get_nnls_coef(
                    np.array(self.model[rsc_type]),
                    np.array(self.actual[rsc_type]))

        for point in self.parameters:
            for k, v in point.items():
                if hasattr(m, k):
                    setattr(m,k,v)
                # hack: ignore fields if module doesn't care

            for rsc_type in RSC_TYPES:
                self.predict[rsc_type].append(Module.rsc(m, self.coef)[rsc_type])

    def get_nnls_coef(self, model, rsc):
        """
        a method for fitting a regression model using
        non-negative least squares. This ensures all
        coefficients are positive.
        """
        nnls = sklearn.linear_model.LinearRegression(
                positive=True, fit_intercept=False)
        return nnls.fit(model, rsc).coef_

    def save_coef(self, outpath):
        for rsc_type in RSC_TYPES:
            filepath = os.path.join(outpath, f"{self.name}_{rsc_type}.npy".lower())
            with open(filepath, "wb") as f:
                np.save(f,self.coef[rsc_type])

    def get_model_error(self):
        for rsc_type in RSC_TYPES:
            diff = np.array(self.predict[rsc_type]) - np.array(self.actual[rsc_type])
            mean_err = diff.mean()
            mse = (diff*diff).mean()
            std = diff.std()

            print("="*5 + "{}_model".format(rsc_type) + "="*5)
            print("Base: mean_err={0}, mse={1}, std={2}".format(mean_err, mse, std))
            print("\n")

    def plot_results(self, outpath):
        for rsc_type in RSC_TYPES:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = self.actual[rsc_type]
            y = self.predict[rsc_type]
            ax.plot(x, x, label="actual")
            ax.scatter(x, y, label="predict", marker="x", color="r")
            ax.set_title(rsc_type)
            ax.set_xlabel("actual")
            ax.set_ylabel("predicted")
            ax.legend()

            filepath = os.path.join(outpath, f"{self.name}_{rsc_type}.jpg".lower())
            fig.savefig(filepath)

