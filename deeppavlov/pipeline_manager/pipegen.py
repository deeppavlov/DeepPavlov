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

from os.path import join
from itertools import product
from copy import deepcopy
from typing import Union, Dict

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.params_search import ParamsSearch


log = get_logger(__name__)


class PipeGen:
    """
    The class implements the generator of standard DeepPavlov configs.
    """
    def __init__(self, config: Union[Dict, str], save_path: str, search: bool = False, search_type: str ='grid', n=10,
                 test_mode=False, cross_val: bool = False):
        """
        Initialize generator with input params.

        Args:
            config: str or dict; path to config file with search pattern, or dict with it config.
            save_path: str; path to folder with pipelines checkpoints.
            search_type: str; random or grid - the trigger that determines type of hypersearch
            n: int; determines the number of generated pipelines, if hyper_search == random.
            test_mode: bool; trigger that determine logic of changing save and loads paths in config.
        """
        if isinstance(config, dict):
            self.main_config = deepcopy(config)
        else:
            self.main_config = deepcopy(read_json(config))

        if 'chainer' not in self.main_config:
            raise ConfigError("Main config file not contain 'chainer' component."
                              "Structure search can not be started without this component.")

        self.dataset_reader = self.main_config.pop("dataset_reader")
        if not isinstance(self.main_config["dataset_iterator"], dict):
            raise ConfigError("Dataset iterator must be one for hole experiment.")
        self.train_config = self.main_config.pop("train")
        self.chainer = self.main_config.pop('chainer')
        self.structure = self.chainer['pipe']

        self.stop_keys = ['in', 'in_x', 'in_y', 'out', 'fit_on', 'name', 'main']
        self.test_mode = test_mode
        self.save_path = save_path
        self.search = search
        self.cross_val = cross_val
        self.search_type = search_type
        self.length = None
        self.pipes = []
        self.N = n

        self._check_component_name()
        self.get_len(deepcopy(self.dataset_reader), deepcopy(self.train_config), deepcopy(self.pipes),
                     deepcopy(self.structure))

        if self.search_type not in ['grid', 'random']:
            raise ValueError("Sorry {0} search not implemented."
                             " At the moment you can use only 'random' and 'grid' search.".format(self.search_type))

        self.enumerator = self.pipeline_enumeration(self.dataset_reader, self.train_config, self.pipes, self.structure)
        self.generator = self.pipeline_gen()

    def _check_component_name(self) -> None:
        for i, component in enumerate(self.structure):
            for j, example in enumerate(component):
                if example is not None:
                    if "component_name" not in example.keys():
                        raise ConfigError("The pipeline element in config file, on position {0} and with number {1}"
                                          "don't contain the 'component_name' key.".format(i+1, j+1))
        return None

    def get_len(self, dataset_reader, train_config, pipes, structure):
        q = 0
        enumerator = self.pipeline_enumeration(dataset_reader, train_config, pipes, structure)
        for variant_ in enumerator:
            variant_ = list(variant_)
            pipe_var_ = variant_[2:]
            if self.search:
                if self.search_type == 'random':
                    search_gen = self.random_conf_gen(pipe_var_)
                elif self.search_type == 'grid':
                    search_gen = self.grid_conf_gen(pipe_var_)
                else:
                    raise ValueError("Sorry '{0}' search not implemented. "
                                     "At the moment you can use only 'random'"
                                     " and 'grid' search.".format(self.search_type))

                for pipe_ in search_gen:
                    q += 1
                del search_gen
            else:
                q += 1

        del enumerator
        self.length = q

    @staticmethod
    def pipeline_enumeration(dataset_reader, train_config, pipes, structure):
        if isinstance(dataset_reader, list):
            drs = []
            for dr in dataset_reader:
                drs.append(dr)
        else:
            drs = [dataset_reader]

        if 'batch_size' in train_config.keys():
            bs_conf = deepcopy(train_config)
            if isinstance(train_config['batch_size'], list):
                bss = []
                for bs in train_config['batch_size']:
                    bs_conf['batch_size'] = bs
                    bss.append(bs_conf)
            else:
                bss = [train_config]
        else:
            bss = [train_config]

        pipes.append(drs)
        pipes.append(bss)

        for components in structure:
            pipes.append(components)

        return product(*pipes)

    def pipeline_gen(self):
        """
        Generate DeepPavlov standard configs (dicts).
        Returns:
            python generator
        """
        p = 0
        for i, variant in enumerate(self.enumerator):
            variant = list(variant)
            dr_config = variant[0]
            train_config = variant[1]
            pipe_var = variant[2:]
            if self.search:
                if self.search_type == 'random':
                    search_gen = self.random_conf_gen(pipe_var)
                elif self.search_type == 'grid':
                    search_gen = self.grid_conf_gen(pipe_var)
                else:
                    raise ValueError("Sorry '{0}' search not implemented. "
                                     "At the moment you can use only 'random'"
                                     " and 'grid' search.".format(self.search_type))

                for k, pipe in enumerate(search_gen):
                    new_config = deepcopy(self.main_config)
                    new_config['dataset_reader'] = deepcopy(dr_config)
                    new_config['train'] = deepcopy(train_config)
                    new_config['chainer'] = deepcopy(self.chainer)

                    chainer_components = list(pipe)
                    if self.cross_val:
                        new_config['chainer']['pipe'] = chainer_components
                        p += 1
                        yield new_config
                    else:
                        dataset_name = dr_config['data_path']
                        chainer_components = self.change_load_path(chainer_components, p, self.save_path, dataset_name,
                                                                   self.test_mode)
                        new_config['chainer']['pipe'] = chainer_components
                        p += 1
                        yield new_config
            else:
                new_config = deepcopy(self.main_config)
                new_config['dataset_reader'] = deepcopy(dr_config)
                new_config['train'] = deepcopy(train_config)
                new_config['chainer'] = deepcopy(self.chainer)

                if self.cross_val:
                    new_config['chainer']['pipe'] = pipe_var
                    p += 1
                    yield new_config
                else:
                    dataset_name = dr_config['data_path']
                    chainer_components = self.change_load_path(pipe_var, p, self.save_path, dataset_name,
                                                               self.test_mode)
                    new_config['chainer']['pipe'] = chainer_components
                    p += 1
                    yield new_config

    # random generation
    def random_conf_gen(self, pipe_components: list) -> GeneratorExit:
        """
        Creates generator that return all possible pipelines.
        Returns:
            python generator
        """
        sample_gen = ParamsSearch()
        for i in range(self.N):
            new_pipe = []
            for component in pipe_components:
                if component is None:
                    pass
                else:
                    new_component = sample_gen.sample_params(**component)
                    new_pipe.append(new_component)

            yield new_pipe

    def grid_param_gen(self, conf):
        """
        Compute cartesian product of config parameters.
        Args:
            conf: dict; component of search pattern

        Returns:
            list
        """
        search_conf = deepcopy(conf)
        list_of_var = []

        # delete "search" key and element
        del search_conf['search']

        values = list()
        keys = list()

        static_keys = list()
        static_values = list()

        for key, item in search_conf.items():
            if key not in self.stop_keys:
                if isinstance(search_conf[key], list):
                    values.append(item)
                    keys.append(key)
                elif isinstance(search_conf[key], dict):
                    raise ValueError("Grid search is not supported params description by 'dict'.")
                elif isinstance(search_conf[key], tuple):
                    raise ValueError("Grid search is not supported params description by 'tuple'.")
                else:
                    static_values.append(search_conf[key])
                    static_keys.append(key)
            else:
                static_values.append(search_conf[key])
                static_keys.append(key)

        valgen = product(*values)

        config = {}
        for i in range(len(static_keys)):
            config[static_keys[i]] = static_values[i]

        for val in valgen:
            cop = deepcopy(config)
            for i, v in enumerate(val):
                cop[keys[i]] = v
            list_of_var.append(cop)

        return list_of_var

    # grid generation
    def grid_conf_gen(self, pipe_components):
        """
        Creates generator that return all possible pipelines.
        Returns:
            python generator
        """
        def update(el):
            lst = []
            if el is not None:
                if 'search' not in el.keys():
                    lst.append(el)
                else:
                    lst.extend(self.grid_param_gen(el))
            else:
                lst.append(el)
            return lst

        pipes = list()
        for i, x in enumerate(pipe_components):
            ln = []
            for y in x:
                ln.extend(update(y))
            pipes.append(ln)

        lgen = product(*pipes)
        for pipe in lgen:
            pipe = list(pipe)
            for conf in pipe:
                if conf is None:
                    pipe.remove(conf)
            yield pipe

    @staticmethod
    def change_load_path(config, n, save_path, dataset_name, test_mode=False):
        """
        Change save_path and load_path attributes in standard config.
        Args:
            config: dict; the chainer content.
            n: int; pipeline number
            save_path: str; path to root folder where will be saved all checkpoints
            test_mode: bool; trigger that determine a regime of pipeline manager work

        Returns:
            config: dict; new config with changed save and load paths
        """
        for component in config:
            if component.get('main') is True:
                if component.get('save_path', None) is not None:
                    sp = component['save_path'].split('/')[-1]
                    if not test_mode:
                        component['save_path'] = join('..', save_path, dataset_name, 'pipe_{}'.format(n+1), sp)
                    else:
                        component['save_path'] = join('..', save_path, "tmp", dataset_name, 'pipe_{}'.format(n + 1), sp)
                if component.get('load_path', None) is not None:
                    lp = component['load_path'].split('/')[-1]
                    if not test_mode:
                        component['load_path'] = join('..', save_path, dataset_name, 'pipe_{}'.format(n+1), lp)
                    else:
                        component['load_path'] = join('..', save_path, "tmp", dataset_name, 'pipe_{}'.format(n + 1), lp)
            else:
                if component.get('save_path', None) is not None:
                    sp = component['save_path'].split('/')[-1]
                    if not test_mode:
                        component['save_path'] = join('..', save_path, dataset_name, sp)
                    else:
                        component['save_path'] = join('..', save_path, "tmp", dataset_name, sp)
        return config

    def __call__(self, *args, **kwargs):
        return self.generator
