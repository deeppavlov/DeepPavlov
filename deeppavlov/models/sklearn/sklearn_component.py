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

from typing import List, Tuple, Any, Union, Generator, Iterable
import numpy as np
from scipy.sparse import spmatrix
import pickle
from pathlib import Path
from scipy.sparse import issparse, csr_matrix
from scipy.sparse import vstack, hstack
import inspect

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register, cls_from_str
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator

log = get_logger(__name__)


@register("sklearn_component")
class SklearnComponent(Estimator):
    """
    Class implements wrapper for sklearn components for feature extraction,
    feature selection, classification, regression etc.

    Args:
        model_name: string with full name of sklearn model to use, e.g. ``sklearn.linear_model:LogisticRegression``
        save_path: save path for model, e.g. full name ``model_path/model.pkl`` \
            or prefix ``model_path/model`` (still model will be saved to ``model_path/model.pkl``)
        load_path: load path for model, e.g. full name ``model_path/model.pkl`` \
            or prefix ``model_path/model`` (still model will be loaded from ``model_path/model.pkl``)
        infer_method: string name of class method to use for infering model, \
            e.g. ``predict``, ``predict_proba``, ``predict_log_proba``, ``transform``
        kwargs: dictionary with parameters for the sklearn model

    Attributes:
        model: sklearn model instance
        model_name: string with full name of sklearn model to use, e.g. ``sklearn.linear_model:LogisticRegression``
        model_params: dictionary with parameters for the sklearn model without pipe parameters
        pipe_params: dictionary with parameters for pipe: ``in``, ``out``, ``fit_on``, ``main``, ``name``
        save_path: save path for model, e.g. full name ``model_path/model.pkl`` \
            or prefix ``model_path/model`` (still model will be saved to ``model_path/model.pkl``)
        load_path: load path for model, e.g. full name ``model_path/model.pkl`` \
            or prefix ``model_path/model`` (still model will be loaded from ``model_path/model.pkl``)
        infer_method: string name of class method to use for infering model, \
            e.g. ``predict``, ``predict_proba``, ``predict_log_proba``, ``transform``
        epochs_done: number of epochs done
        batches_seen: number of batches seen
        train_examples_seen: number of train examples seen
    """
    def __init__(self, model_name: str,
                 save_path: Union[str, Path] = None,
                 load_path: Union[str, Path] = None,
                 infer_method: str = "predict", **kwargs) -> None:
        """
        Initialize component with given parameters
        """

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
        """
        Fit model on the given data

        Args:
            *args: list of x-inputs and, optionally, one y-input (the last one) to fit on.
                Possible input (x0, ..., xK, y) or (x0, ..., xK) '
                where K is the number of input data elements (the length of list ``in`` from config). \
                In case of several inputs (K > 1) input features will be stacked. \
                For example, one has x0: (n_samples, n_features0), ..., xK: (n_samples, n_featuresK), \
                then model will be trained on x: (n_samples, n_features0 + ... + n_featuresK).
            **kwargs: additional parameters

        Returns:
            None
        """
        n_inputs = len(self.pipe_params["in"]) if isinstance(self.pipe_params["in"], list) else 1
        x_features = self.compose_input_data(args[:n_inputs])
        if len(args) > n_inputs:
            y_ = np.squeeze(np.array(args[-1]))
        else:
            y_ = None

        try:
            log.info("Fitting model {}".format(self.model_name))
            self.model.fit(x_features, y_)
        except TypeError or ValueError:
            try:
                if issparse(x_features):
                    log.info("Converting input for model {} to dense array".format(self.model_name))
                    self.model.fit(x_features.todense(), y_)
                else:
                    log.info("Converting input for model {} to sparse array".format(self.model_name))
                    self.model.fit(csr_matrix(x_features), y_)
            except:
                raise ConfigError("Can not fit on the given data".format(self.model_name))

        return

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        Infer on the given data according to given in the config infer method, \
            e.g. ``"predict", "predict_proba", "transform"``

        Args:
            *args: list of inputs
            **kwargs: additional arguments

        Returns:
            predictions, e.g. list of labels, array of probability distribution, sparse array of vectorized samples
        """
        predictions = self.infer_on_batch(args)
        return predictions

    def infer_on_batch(self, x):
        """
        Infer on the given data according to given in the config infer method, \
            e.g. ``"predict", "predict_proba", "transform"``

        Args:
            *args: list of inputs
            **kwargs: additional arguments

        Returns:
            predictions, e.g. list of labels, array of probability distribution, sparse array of vectorized samples
        """
        x_features = self.compose_input_data(x)

        try:
            predictions = getattr(self.model, self.infer_method)(x_features)
        except:
            try:
                if issparse(x_features):
                    log.info("Converting input for model {} to dense array".format(self.model_name))
                    predictions = getattr(self.model, self.infer_method)(x_features.todense())
                else:
                    log.info("Converting input for model {} to sparse array".format(self.model_name))
                    predictions = getattr(self.model, self.infer_method)(csr_matrix(x_features))
            except:
                raise ConfigError("Can not infer on the given data".format(self.model_name))

        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def init_from_scratch(self) -> None:
        """
        Initialize ``self.model`` as some sklearn model from scratch with given in ``self.model_params`` parameters.

        Returns:
            None
        """
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

    def load(self, fname: str = None) -> None:
        """
        Initialize ``self.model`` as some sklearn model from saved re-initializing ``self.model_params`` parameters. \
            If in new given parameters ``warm_start`` is set to True and given model admits ``warm_start`` parameter, \
            model will be initilized from saved with opportunity to continue fitting.

        Args:
            fname: string name of path to model to load from

        Returns:
            None
        """
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
            log.info("Model {} loaded  with parameters".format(self.model_name))

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
        """
        Save ``self.model`` to the file from ``fname`` or, if not given, ``self.save_path``. \
            If ``self.save_path`` does not have ``.pkl`` extension, then it will be replaced \
            to ``str(Path(self.save_path).stem) + ".pkl"``

        Args:
            fname:  string name of path to model to save to

        Returns:
            None
        """
        if fname is None:
            fname = self.save_path

        if Path(fname).suffix != ".pkl":
            fname = str(Path(fname).stem) + ".pkl"

        log.info("Saving model to {}".format(fname))
        with open(fname, "wb") as f:
            pickle.dump(self.model, f)
        return

    @staticmethod
    def compose_input_data(x: List[Union[Tuple[Union[np.ndarray, list, spmatrix, str]],
                                         List[Union[np.ndarray, list, spmatrix, str]],
                                         np.ndarray, spmatrix]]) -> Union[spmatrix, np.ndarray]:
        """
        Stack given list of different types of inputs to the one matrix. If one of the inputs is a sparse matrix, \
            then output will be also a sparse matrix

        Args:
            x: list of data elements

        Returns:
            sparse or dense array of stacked data
        """
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
        """
        Delete ``self.model`` from memory

        Returns:
            None
        """
        del self.model
        return

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
        """
        Get list of names of given function's parameters

        Args:
            f: function

        Returns:
            list of names of given function's parameters
        """
        return inspect.getfullargspec(f)[0]

    @staticmethod
    def get_class_attributes(cls) -> List[str]:
        """
        Get list of names of given class' attributes

        Args:
            cls: class

        Returns:
            list of names of given class' attributes
        """
        return list(cls.__dict__.keys())
