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

from os.path import join
from itertools import product
from copy import deepcopy
from typing import Union, Dict, Generator

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.params_search import ParamsSearch


log = get_logger(__name__)


class PipeGen:
    """
    The class implements the generator of standard DeepPavlov configs.
    """

    def __init__(self, config: Union[Dict, str], save_path: str, sample_num=10, test_mode=False,
                 cross_val: bool = False):
        """
        Initialize generator with input params.

        Args:
            config: str or dict; path to config file with search pattern, or dict with it config.
            save_path: str; path to folder with pipelines checkpoints.
            sample_num: int; determines the number of generated pipelines, if hyper_search == random.
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

        self.test_mode = test_mode
        self.save_path = save_path
        self.cross_val = cross_val
        self.length = None
        self.pipes = []
        self.N = sample_num

        self._check_component_name()
        self.get_len()
        self.enumerator = self.pipeline_enumeration()
        self.generator = self.pipeline_gen()

    def _check_component_name(self) -> None:
        for i, component in enumerate(self.structure):
            for j, example in enumerate(component):
                if example is not None:
                    if "component_name" not in example.keys():
                        raise ConfigError("The pipeline element in config file, on position {0} and with number {1}"
                                          "don't contain the 'component_name' key.".format(i + 1, j + 1))
        return None

    def get_len(self):
        self.enumerator = self.pipeline_enumeration()
        generator = self.pipeline_gen()
        self.length = len(list(generator))
        self.pipes = []
        del generator

    def __len__(self):
        return self.length

    def pipeline_enumeration(self):
        if isinstance(self.dataset_reader, list):
            drs = []
            for dr in self.dataset_reader:
                drs.append(dr)
        else:
            drs = [self.dataset_reader]

        if 'batch_size' in self.train_config.keys():
            bs_conf = deepcopy(self.train_config)
            if isinstance(self.train_config['batch_size'], list):
                bss = []
                for bs in self.train_config['batch_size']:
                    bs_conf['batch_size'] = bs
                    bss.append(bs_conf)
            else:
                bss = [self.train_config]
        else:
            bss = [self.train_config]

        self.pipes.append(drs)
        self.pipes.append(bss)

        for components in self.structure:
            self.pipes.append(components)

        return product(*self.pipes)

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

            for grid_pipe in self.grid_conf_gen(pipe_var):
                grid_pipe = list(grid_pipe)
                for pipe in self.random_conf_gen(grid_pipe):

                    new_config = dict(dataset_reader=deepcopy(dr_config),
                                      dataset_iterator=self.main_config['dataset_iterator'],
                                      chainer=self.chainer, train=train_config)
                    if 'metadata' in self.main_config.keys():
                        new_config['metadata'] = self.main_config['metadata']

                    chainer_components = list(pipe)
                    if not self.cross_val:
                        dataset_name = dr_config['data_path']
                        chainer_components = self.change_load_path(chainer_components, p, self.save_path, dataset_name,
                                                                   self.test_mode)
                        new_config['chainer']['pipe'] = chainer_components
                        p += 1
                        yield new_config
                    else:
                        new_config['chainer']['pipe'] = chainer_components
                        yield new_config

    # random generation
    def random_conf_gen(self, pipe_components: list) -> Generator:
        """
        Creates generator that return all possible pipelines.
        Returns:
            python generator
        """
        sample_gen = ParamsSearch()
        new_pipes = []
        for component in pipe_components:
            new_components = []
            if component:
                search = False
                for key, item in component.items():
                    if isinstance(item, dict):
                        for item_key in item.keys():
                            if item_key.startswith('search_'):
                                search = True
                                break

                if search:
                    for i in range(self.N):
                        new_components.append(sample_gen.sample_params(**component))
                    new_pipes.append(new_components)
                else:
                    new_components.append(component)
                    new_pipes.append(new_components)
            else:
                pass

        for new_config in product(*new_pipes):
            yield new_config

    # grid generation
    @staticmethod
    def grid_conf_gen(pipe_components: list) -> Generator:
        """
        Compute cartesian product of config parameters.
        Args:
            pipe_components: list of dicts; config if components

        Returns:
            list
        """
        list_of_variants = []
        # find in config keys for grid search
        for i, component in enumerate(pipe_components):
            if component is not None:
                for key, item in component.items():
                    if isinstance(item, dict):
                        if 'search_grid' in item.keys():
                            var_list = list()
                            for var in item['search_grid']:
                                var_dict = dict()
                                var_dict[var] = [i, key]
                                var_list.append(var_dict)
                            list_of_variants.append(var_list)
            else:
                pass
        # create generator
        valgen = product(*list_of_variants)
        # run generator
        for variant in valgen:
            search_conf = deepcopy(pipe_components)
            for val in variant:
                for value, item in val.items():
                    search_conf[item[0]][item[1]] = value
            yield search_conf

    @staticmethod
    def change_load_path(config, n, save_path, dataset_name, test_mode=False):
        """
        Change save_path and load_path attributes in standard config.
        Args:
            config: dict; the chainer content.
            n: int; pipeline number
            save_path: str; path to root folder where will be saved all checkpoints
            dataset_name: str; name of dataset
            test_mode: bool; trigger that determine a regime of pipeline manager work

        Returns:
            config: dict; new config with changed save and load paths
        """
        for component in config:
            if component.get('main') is True:
                if component.get('save_path', None) is not None:
                    sp = component['save_path'].split('/')[-1]
                    if not test_mode:
                        component['save_path'] = join('..', save_path, dataset_name, 'pipe_{}'.format(n + 1), sp)
                    else:
                        component['save_path'] = join('..', save_path, "tmp", dataset_name, 'pipe_{}'.format(n + 1), sp)
                if component.get('load_path', None) is not None:
                    lp = component['load_path'].split('/')[-1]
                    if not test_mode:
                        component['load_path'] = join('..', save_path, dataset_name, 'pipe_{}'.format(n + 1), lp)
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
