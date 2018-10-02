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
from scipy.sparse import vstack
import inspect
from scipy.sparse import csr_matrix
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.multiclass import *
from sklearn.multioutput import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.tree import *
from sklearn.semi_supervised import *
from sklearn.exceptions import NotFittedError

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator

log = get_logger(__name__)


@register("sklearn_classifier")
class SklearnClassier(Estimator):
    def __init__(self, model_name: str, classes: Union[list, np.ndarray, Generator],
                 save_path: Union[str, Path] = None,
                 load_path: Union[str, Path] = None,
                 infer_method: str = "predict", **kwargs) -> None:
        super().__init__(save_path=save_path, load_path=load_path, **kwargs)
        self.model = self.load(model_name=model_name, **kwargs)
        self.infer_method = infer_method
        self.epochs_done = 0
        self.batches_seen = 0
        self.train_examples_seen = 0
        self.classes = list(np.sort(np.array(list(classes))))
        self.n_classes = len(self.classes)

    def fit(self, x: Union[List[List[float]], np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
            y: Union[np.ndarray, List[list]], *args, **kwargs) -> None:
        if len(x) != 0:
            if isinstance(x[0], csr_matrix):
                x_features = vstack(list(x))
            elif isinstance(x[0], np.ndarray):
                x_features = np.vstack(list(x))
            elif isinstance(x, list) and isinstance(x[0], list):
                x_features = x
            elif isinstance(x, np.ndarray):
                x_features = x
            else:
                ConfigError('Not implemented this type of vectors')
        else:
            ConfigError("Input vectors cannot be empty")

        given_params = {}
        if kwargs:
            available_params = self.get_function_params(self.model.fit)
            for param_name in kwargs.keys():
                if param_name in available_params:
                    given_params[param_name] = kwargs[param_name]

        y_ = np.array(y).reshape(-1, 1)
        self.model.fit(x_features, y_, **given_params)
        return

    def __call__(self, x, **kwargs) -> np.ndarray:
        predictions = self.infer_on_batch(x, **kwargs)
        return predictions

    def infer_on_batch(self, x, **kwargs):
        if len(x) != 0:
            if isinstance(x[0], csr_matrix):
                x_features = vstack(list(x))
            elif isinstance(x[0], np.ndarray):
                x_features = np.vstack(list(x))
            elif isinstance(x, list) and isinstance(x[0], list):
                x_features = x
            elif isinstance(x, np.ndarray):
                x_features = x
            else:
                ConfigError('Not implemented this type of vectors')
        else:
            ConfigError("Input vectors cannot be empty")
        try:
            predictions = getattr(self.model, self.infer_method)(x_features)
        except NotFittedError:
            x_random = np.random.random(size=(100, len(x[0])))
            y_random = np.zeros((100, self.n_classes))
            self.model.fit(x_random, y_random)
            predictions = getattr(self.model, self.infer_method)(x_features)

        return predictions

    def init_from_scratch(self, model_name: str, **kwargs) -> Any:
        log.warning("Initializing model {} from scratch".format(model_name))
        model_function = globals().get(model_name, None)

        if model_function is None:
            ConfigError("Model with {} model_name was not found".format(model_name))

        given_params = {}
        if kwargs:
            available_params = self.get_function_params(model_function)
            for param_name in kwargs.keys():
                if param_name in available_params:
                    given_params[param_name] = kwargs[param_name]

        model = model_function(**given_params)
        return model

    def load(self, fname: str = None,
             model_name: str = None, **kwargs) -> Any:
        if fname is None:
            fname = self.load_path

        if Path(fname).suffix != ".pkl":
            fname = str(Path(fname).stem) + ".pkl"

        if fname.exists():
            log.warning("Loading model {} from {}".format(model_name, fname))
            with open(fname, "rb") as f:
                model = pickle.load(f)
        else:
            log.warning("Cannot load model from {}".format(fname))
            model = self.init_from_scratch(model_name=model_name, **kwargs)

        return model

    def save(self, fname: str = None) -> None:
        if fname is None:
            fname = self.save_path

        if Path(fname).suffix != ".pkl":
            fname = str(Path(fname).stem) + ".pkl"

        log.warning("Saving model to {}".format(fname))
        with open(fname, "wb") as f:
            pickle.dump(self.model, f)
        return

    def reset(self) -> None:
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
