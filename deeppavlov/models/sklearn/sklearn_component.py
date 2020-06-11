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

import inspect
import pickle
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union, Callable

import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.sparse import spmatrix
from scipy.sparse import vstack, hstack

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register, cls_from_str
from deeppavlov.core.models.estimator import Estimator

log = getLogger(__name__)


@register("sklearn_component")
class SklearnComponent(Estimator):
    """
    Class implements wrapper for sklearn components for feature extraction,
    feature selection, classification, regression etc.

    Args:
        model_class: string with full name of sklearn model to use, e.g. ``sklearn.linear_model:LogisticRegression``
        save_path: save path for model, e.g. full name ``model_path/model.pkl`` \
            or prefix ``model_path/model`` (still model will be saved to ``model_path/model.pkl``)
        load_path: load path for model, e.g. full name ``model_path/model.pkl`` \
            or prefix ``model_path/model`` (still model will be loaded from ``model_path/model.pkl``)
        infer_method: string name of class method to use for infering model, \
            e.g. ``predict``, ``predict_proba``, ``predict_log_proba``, ``transform``
        ensure_list_output: whether to ensure that output for each sample is iterable (but not string)
        kwargs: dictionary with parameters for the sklearn model

    Attributes:
        model: sklearn model instance
        model_class: string with full name of sklearn model to use, e.g. ``sklearn.linear_model:LogisticRegression``
        model_params: dictionary with parameters for the sklearn model without pipe parameters
        pipe_params: dictionary with parameters for pipe: ``in``, ``out``, ``fit_on``, ``main``, ``name``
        save_path: save path for model, e.g. full name ``model_path/model.pkl`` \
            or prefix ``model_path/model`` (still model will be saved to ``model_path/model.pkl``)
        load_path: load path for model, e.g. full name ``model_path/model.pkl`` \
            or prefix ``model_path/model`` (still model will be loaded from ``model_path/model.pkl``)
        infer_method: string name of class method to use for infering model, \
            e.g. ``predict``, ``predict_proba``, ``predict_log_proba``, ``transform``
        ensure_list_output: whether to ensure that output for each sample is iterable (but not string)
    """

    def __init__(self, model_class: str,
                 save_path: Union[str, Path] = None,
                 load_path: Union[str, Path] = None,
                 infer_method: str = "predict",
                 ensure_list_output: bool = False,
                 **kwargs) -> None:
        """
        Initialize component with given parameters
        """

        super().__init__(save_path=save_path, load_path=load_path, **kwargs)
        self.model_class = model_class
        self.model_params = kwargs
        self.model = None
        self.ensure_list_output = ensure_list_output
        self.pipe_params = {}
        for required in ["in", "out", "fit_on", "main", "name"]:
            self.pipe_params[required] = self.model_params.pop(required, None)

        self.load()
        self.infer_method = getattr(self.model, infer_method)

    def fit(self, *args) -> None:
        """
        Fit model on the given data

        Args:
            *args: list of x-inputs and, optionally, one y-input (the last one) to fit on.
                Possible input (x0, ..., xK, y) or (x0, ..., xK) '
                where K is the number of input data elements (the length of list ``in`` from config). \
                In case of several inputs (K > 1) input features will be stacked. \
                For example, one has x0: (n_samples, n_features0), ..., xK: (n_samples, n_featuresK), \
                then model will be trained on x: (n_samples, n_features0 + ... + n_featuresK).

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
            log.info("Fitting model {}".format(self.model_class))
            self.model.fit(x_features, y_)
        except TypeError or ValueError:
            if issparse(x_features):
                log.info("Converting input for model {} to dense array".format(self.model_class))
                self.model.fit(x_features.todense(), y_)
            else:
                log.info("Converting input for model {} to sparse array".format(self.model_class))
                self.model.fit(csr_matrix(x_features), y_)

        return

    def __call__(self, *args):
        """
        Infer on the given data according to given in the config infer method, \
            e.g. ``"predict", "predict_proba", "transform"``

        Args:
            *args: list of inputs

        Returns:
            predictions, e.g. list of labels, array of probability distribution, sparse array of vectorized samples
        """
        x_features = self.compose_input_data(args)

        try:
            predictions = self.infer_method(x_features)
        except TypeError or ValueError:
            if issparse(x_features):
                log.info("Converting input for model {} to dense array".format(self.model_class))
                predictions = self.infer_method(x_features.todense())
            else:
                log.info("Converting input for model {} to sparse array".format(self.model_class))
                predictions = self.infer_method(csr_matrix(x_features))

        if isinstance(predictions, list):
            #  ``predict_proba`` sometimes returns list of n_outputs (each output corresponds to a label)
            #  but we will return (n_samples, n_labels)
            #  where each value is a probability of a sample to belong with the label
            predictions_ = [[predictions[j][i][1] for j in range(len(predictions))] for i in range(x_features.shape[0])]
            predictions = np.array(predictions_)

        if self.ensure_list_output and len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)

        if issparse(predictions):
            return predictions
        else:
            return predictions.tolist()

    def init_from_scratch(self) -> None:
        """
        Initialize ``self.model`` as some sklearn model from scratch with given in ``self.model_params`` parameters.

        Returns:
            None
        """
        log.info("Initializing model {} from scratch".format(self.model_class))
        model_function = cls_from_str(self.model_class)

        if model_function is None:
            raise ConfigError("Model with {} model_class was not found.".format(self.model_class))

        given_params = {}
        if self.model_params:
            available_params = self.get_function_params(model_function)
            for param_name in self.model_params.keys():
                if param_name in available_params:
                    try:
                        given_params[param_name] = cls_from_str(self.model_params[param_name])
                    except (AttributeError, ValueError, ConfigError):
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

        fname = Path(fname).with_suffix('.pkl')

        if fname.exists():
            log.info("Loading model {} from {}".format(self.model_class, str(fname)))
            with open(fname, "rb") as f:
                self.model = pickle.load(f)

            warm_start = self.model_params.get("warm_start", None)
            self.model_params = {param: getattr(self.model, param) for param in self.get_class_attributes(self.model)}
            self.model_class = self.model.__module__ + self.model.__class__.__name__
            log.info("Model {} loaded  with parameters".format(self.model_class))

            if warm_start and "warm_start" in self.model_params.keys():
                self.model_params["warm_start"] = True
                log.info("Fitting of loaded model can be continued because `warm_start` is set to True")
            else:
                log.warning("Fitting of loaded model can not be continued. Model can be fitted from scratch."
                            "If one needs to continue fitting, please, look at `warm_start` parameter")
        else:
            log.warning("Cannot load model from {}".format(str(fname)))
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

        fname = Path(fname).with_suffix('.pkl')

        log.info("Saving model to {}".format(str(fname)))
        with open(fname, "wb") as f:
            pickle.dump(self.model, f, protocol=4)
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

    @staticmethod
    def get_function_params(f: Callable) -> List[str]:
        """
        Get list of names of given function's parameters

        Args:
            f: function

        Returns:
            list of names of given function's parameters
        """
        return inspect.getfullargspec(f)[0]

    @staticmethod
    def get_class_attributes(cls: type) -> List[str]:
        """
        Get list of names of given class' attributes

        Args:
            cls: class

        Returns:
            list of names of given class' attributes
        """
        return list(cls.__dict__.keys())
