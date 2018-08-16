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

import numpy as np
from copy import deepcopy
import random
from typing import List, Generator, Tuple, Any

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('params_search')
class ParamsSearch:
    """
    Class determine the main operations for parameters search
    like finding all changing parameters.

    Args:
        prefix: prefix to determine special keys like `PREFIX_range`, `PREFIX_bool`, `PREFIX_choice`
        seed: random seed for initialization
        **kwargs: basic config with parameters

    Attributes:
        basic_config: dictionary with initial evolutionary config
        prefix: prefix to determine special keys like `PREFIX_range`, `PREFIX_bool`, `PREFIX_choice`
        paths_to_params: list of lists of keys and/or integers (for list)
                with relative paths to searched parameters
        n_params: number of searched parameters
        eps: EPS value
        paths_to_fiton_dicts: list of lists of keys and/or integers (for list)
                with relative paths to dictionaries that can be "fitted on"
        n_fiton_dicts: number of dictionaries that can be "fitted on"
    """

    def __init__(self,
                 prefix="search",
                 seed: int = None,
                 **kwargs):
        """
        Initialize evolution with random population
        """

        self.basic_config = deepcopy(kwargs)
        self.prefix = prefix

        self.paths_to_params = []
        for search_type in [prefix + "_range", prefix + "_choice", prefix + "_bool"]:
            for path_ in self.find_model_path(self.basic_config, search_type):
                self.paths_to_params.append(path_)

        self.n_params = len(self.paths_to_params)

        self.eps = 1e-6

        self.paths_to_fiton_dicts = []
        for path_ in self.find_model_path(self.basic_config, "fit_on"):
            self.paths_to_fiton_dicts.append(path_)
        self.n_fiton_dicts = len(self.paths_to_fiton_dicts)

        if seed is None:
            pass
        else:
            np.random.seed(seed)
            random.seed(seed)

    def find_model_path(self, config: dict, key_model: str, path: list = []) -> Generator:
        """
        Find paths to all dictionaries in config that contain key 'key_model'

        Args:
            config: dictionary
            key_model: key of sub-dictionary to be found
            path: list of keys and/or integers (for list) with relative path (needed for recursion)

        Returns:
            path in config -- list of keys (strings and integers)
        """
        config_pointer = config
        if type(config_pointer) is dict and key_model in config_pointer.keys():
            yield path
        else:
            if type(config_pointer) is dict:
                for key in list(config_pointer.keys()):
                    for path_ in self.find_model_path(config_pointer[key], key_model, path + [key]):
                        yield path_
            elif type(config_pointer) is list:
                for i in range(len(config_pointer)):
                    for path_ in self.find_model_path(config_pointer[i], key_model, path + [i]):
                        yield path_

    @staticmethod
    def insert_value_or_dict_into_config(config: dict, path: list,
                                         value: [int, float, str, bool, list, dict, np.ndarray]) -> dict:
        """
        Insert value to dictionary determined by path[:-1] in field with key path[-1]

        Args:
            config: dictionary
            path: list of keys and/or integers (for list)
            value: value to be inserted

        Returns:
            config with inserted value
        """
        config_copy = deepcopy(config)
        config_pointer = config_copy
        for el in path[:-1]:
            if type(config_pointer) is dict:
                config_pointer = config_pointer.setdefault(el, {})
            elif type(config_pointer) is list:
                config_pointer = config_pointer[el]
            else:
                pass
        config_pointer[path[-1]] = value
        return config_copy

    @staticmethod
    def get_value_from_config(config: dict, path: list) -> Any:
        """
        Return value of config element determined by path

        Args:
            config: dictionary
            path: list of keys and/or integers (for list)

        Returns:
            value
        """
        config_copy = deepcopy(config)
        config_pointer = config_copy
        for el in path[:-1]:
            if type(config_pointer) is dict:
                config_pointer = config_pointer.setdefault(el, {})
            elif type(config_pointer) is list:
                config_pointer = config_pointer[el]
            else:
                pass
        return config_pointer[path[-1]]

    def initialize_params_in_config(self, basic_config: dict, paths: List[list]) -> dict:
        """
        Randomly initialize all the changable parameters in config

        Args:
            basic_config: config where changable parameters are dictionaries with keys
                `evolve_range`, `evolve_bool`, `evolve_choice`
            paths: list of paths to changable parameters

        Returns:
            config
        """
        config = deepcopy(basic_config)
        for path_ in paths:
            param_name = path_[-1]
            value = self.get_value_from_config(basic_config, path_)
            if type(value) is dict:
                if (value.get(self.prefix + "_choice") or
                        value.get(self.prefix + "_range") or
                        value.get(self.prefix + "_bool")):
                    config = self.insert_value_or_dict_into_config(
                        config, path_,
                        self.sample_params(**{param_name: deepcopy(value)})[param_name])

        return config

    def sample_params(self, **params) -> dict:
        """
        Sample parameters according to the given possible values

        Args:
            **params: dictionary like {"param_0": {"evolve_range": [0, 10]},
                                       "param_1": {"evolve_range": [0, 10], "discrete": true},
                                       "param_2": {"evolve_range": [0, 1], "scale": "log"},
                                       "param_3": {"evolve_bool": true},
                                       "param_4": {"evolve_choice": [0, 1, 2, 3]}}

        Returns:
            dictionary with randomly sampled parameters
        """
        if not params:
            return {}
        else:
            params_copy = deepcopy(params)
        params_sample = dict()
        for param, param_val in params_copy.items():
            if isinstance(param_val, dict):
                if self.prefix + '_bool' in param_val and param_val[self.prefix + '_bool']:
                    sample = bool(random.choice([True, False]))
                elif self.prefix + '_range' in param_val:
                    sample = self._sample_from_ranges(param_val)
                elif self.prefix + '_choice' in param_val:
                    sample = random.choice(param_val[self.prefix + '_choice'])
                else:
                    sample = param_val
                params_sample[param] = sample
            else:
                params_sample[param] = params_copy[param]
        return params_sample

    def _sample_from_ranges(self, opts: dict) -> [int, float]:
        """
        Sample parameters from ranges

        Args:
            opts: dictionary  {"evolve_range": [0, 10]} or \
                              {"evolve_range": [0, 10], "discrete": true} or \
                              {"evolve_range": [0, 1], "scale": "log"}

        Returns:
            random parameter value from range
        """
        from_ = opts[self.prefix + '_range'][0]
        to_ = opts[self.prefix + '_range'][1]
        if opts.get('scale', None) == 'log':
            sample = self._sample_log(from_, to_)
        else:
            sample = np.random.uniform(from_, to_)
        if opts.get('discrete', False):
            sample = int(np.round(sample))
        return sample

    @staticmethod
    def _sample_log(from_: float = 0., to_: float = 1.) -> float:
        """
        Sample parameters from ranges with log scale

        Args:
            from_: lower boundary of values
            to_:  upper boundary of values

        Returns:
            random parameters value from range with log scale
        """
        sample = np.exp(np.random.uniform(np.log(from_), np.log(to_)))
        return float(sample)
