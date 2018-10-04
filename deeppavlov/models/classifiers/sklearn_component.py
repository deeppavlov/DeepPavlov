# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Any, Union, Generator
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import issparse
from scipy.sparse import vstack, hstack
import inspect

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register, cls_from_str
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator

log = get_logger(__name__)


@register("sklearn_component")
class SklearnComponent(Estimator):
    def __init__(self, model_name: str,
                 save_path: Union[str, Path] = None,
                 load_path: Union[str, Path] = None,
                 infer_method: str = "predict", **kwargs) -> None:

        super().__init__(save_path=save_path, load_path=load_path, **kwargs)
        self.model_name = model_name
        self.model_params = kwargs
        self.model = None
        self.pipe_params = {}
        for required in ["in", "out", "fit_on", "main", "name"]:
            self.pipe_params[required] = self.model_params.pop(required, None)

        self.load()
        self.infer_method = infer_method
        self.epochs_done = 0
        self.batches_seen = 0
        self.train_examples_seen = 0

    def fit(self, *args, **kwargs) -> None:
        x_features = self.compose_input_data(args[:-1])
        y_ = np.squeeze(np.array(args[-1]))

        try:
            log.info("Fitting model {}".format(self.model_name))
            self.model.fit(x_features, y_)
        except ValueError:
            raise ConfigError("Incompatible dimensions, check parameters of model. "
                              "Got X of shape {}, y of shape {}".format(x_features.shape, y_.shape))
        return

    def __call__(self, *args, **kwargs) -> np.ndarray:
        predictions = self.infer_on_batch(args)
        return predictions

    def infer_on_batch(self, x):
        x_features = self.compose_input_data(x)
        predictions = getattr(self.model, self.infer_method)(x_features)

        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def init_from_scratch(self) -> Any:
        log.info("Initializing model {} from scratch".format(self.model_name))
        model_function = cls_from_str(self.model_name)

        if model_function is None:
            raise ConfigError("Model with {} model_name was not found.".format(self.model_name))

        given_params = {}
        if self.model_params:
            available_params = self.get_function_params(model_function)
            for param_name in self.model_params.keys():
                if param_name in available_params:
                    try:
                        given_params[param_name] = cls_from_str(self.model_params[param_name])
                    except:
                        given_params[param_name] = self.model_params[param_name]

        self.model = model_function(**given_params)
        return

    def load(self, fname: str = None) -> Any:
        if fname is None:
            fname = self.load_path

        if Path(fname).suffix != ".pkl":
            fname = str(Path(fname).stem) + ".pkl"

        if fname.exists():
            log.info("Loading model {} from {}".format(self.model_name, fname))
            with open(fname, "rb") as f:
                self.model = pickle.load(f)

            warm_start = self.model_params.get("warm_start", None)
            self.model_params = {param: getattr(self.model, param) for param in self.get_class_attributes(self.model)}
            self.model_name = self.model.__module__ + self.model.__class__.__name__
            log.info("Model {} loaded  with parameters: {}".format(self.model_name, self.model_params))

            if warm_start and "warm_start" in self.model_params.keys():
                self.model_params["warm_start"] = True
                log.info("Fitting of loaded model can be continued because `warm_start` is set to True")
            else:
                log.warning("Fitting of loaded model can not be continued. Model can be fitted from scratch."
                            "If one needs to continue fitting, please, look at `warm_start` parameter")
        else:
            log.warning("Cannot load model from {}".format(fname))
            self.init_from_scratch()

        return

    def save(self, fname: str = None) -> None:
        if fname is None:
            fname = self.save_path

        if Path(fname).suffix != ".pkl":
            fname = str(Path(fname).stem) + ".pkl"

        log.info("Saving model to {}".format(fname))
        with open(fname, "wb") as f:
            pickle.dump(self.model, f)
        return

    def compose_input_data(self, x):
        x_features = []
        for i in range(len(x)):
            if ((isinstance(x[i], tuple) or isinstance(x[i], list) or isinstance(x[i], np.ndarray) and len(x[i]))
                    or (issparse(x[i]) and x[i].shape[0])):
                if issparse(x[i][0]):
                    x_features.append(vstack(list(x[i])))
                elif isinstance(x[i][0], np.ndarray) or isinstance(x[i][0], list):
                    x_features.append(np.vstack(list(x[i])))
                elif isinstance(x[i][0], str):
                    x_features.append(np.array(x[i]))
                else:
                    raise ConfigError('Not implemented this type of vectors')
            else:
                raise ConfigError("Input vectors cannot be empty")

        sparse = False
        for inp in x_features:
            if issparse(inp):
                sparse = True
        if sparse:
            x_features = hstack(list(x_features))
        else:
            x_features = np.hstack(list(x_features))

        return x_features

    def destroy(self) -> None:
        del self.model

    def process_event(self, event_name: str, data: dict):
        """
        Process event after epoch
        Args:
            event_name: whether event is send after epoch or batch.
                    Set of values: ``"after_epoch", "after_batch"``
            data: event data (dictionary)

        Returns:
            None
        """
        if event_name == "after_epoch":
            self.epochs_done = data["epochs_done"]
            self.batches_seen = data["batches_seen"]
            self.train_examples_seen = data["train_examples_seen"]
        return

    @staticmethod
    def get_function_params(f) -> List[str]:
        return inspect.getfullargspec(f)[0]

    @staticmethod
    def get_class_attributes(cls) -> List[str]:
        return list(cls.__dict__.keys())
