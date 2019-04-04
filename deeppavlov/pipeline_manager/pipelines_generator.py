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

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple
from typing import Union

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.params_search import ParamsSearch


class PipeGen:
    """
    The :class:`~pipeline_manager.pipegen.PipeGen` implements the function of generator of standard DeepPavlov configs.
    Based on the input config, the generator creates a set of pipelines, as well as variants of the same pipeline
    with a different set of hyperparameters using the "random" or "grid" search. Also in all generated configs the save
    and load paths change to intermediate ones.

    Args:
            config: path to config file with search pattern, or dict with it config.
            save_path: path to folder with pipelines checkpoints.
            mode: working mode of generator, can be 'random' or 'grid'.
            sample_num: determines the number of generated pipelines, if hyper_search == random.
    """

    def __init__(self,
                 config: Union[str, Dict, Path],
                 save_path: Union[str, Path],
                 mode: str = 'random',
                 sample_num: int = 10) -> None:
        """ Initialize generator with input params. """
        if mode in ['random', 'grid']:
            self.search_mode = mode
        else:
            raise ConfigError(f"'{mode} search' not implemented. Only 'grid' and 'random' search are available.")

        self.save_path = save_path
        self.N = sample_num
        self.pipe_ind = None
        self.pipes = []

        if not isinstance(config, dict):
            config = read_json(config)

        self.tmp_config = deepcopy(config)
        self.structure = config['chainer']['pipe']

    def modify_config(self, pipe: tuple):
        chainer_components = self.change_load_path(list(pipe), self.pipe_ind, self.save_path)
        self.tmp_config['chainer']['pipe'] = chainer_components
        return deepcopy(self.tmp_config)

    def __call__(self) -> Generator:
        """
        Creates a configs with a different set of hyperparameters based on the primary set of pipelines.

        Returns:
            iterator of final sets of configs (dicts)
        """
        self.pipe_ind = 0
        if self.search_mode == 'random':
            gen_method = self.random_conf_gen
        else:
            gen_method = self.grid_conf_gen

        for components in self.structure:
            if isinstance(components, dict):
                self.pipes.append([components])
            else:
                self.pipes.append(components)

        for pipe_var in product(*self.pipes):
            for pipe_ in gen_method(pipe_var):
                self.pipe_ind += 1
                yield self.modify_config(pipe_)

    # random generation
    def random_conf_gen(self, pipe_components: Tuple[dict]) -> Generator:
        """
        Creates a set of configs with a different set of hyperparameters using "random search".

        Args:
            pipe_components: list of dicts; config if components

        Returns:
            random search iterator
        """
        sample_gen = ParamsSearch(prefix="random", seed=42)
        new_pipes = []
        for component in pipe_components:
            new_components = []
            if component:
                search = any(item_key.startswith('random_')
                             for item in component.values() if isinstance(item, dict)
                             for item_key in item.keys())
                if search:
                    for i in range(self.N):
                        new_components.append(sample_gen.sample_params(**component))
                else:
                    new_components.append(component)
                new_pipes.append(new_components)

        for new_config in product(*new_pipes):
            yield new_config

    # grid generation
    @staticmethod
    def grid_conf_gen(pipe_components: Tuple[dict]) -> Generator:
        """
        Creates a set of configs with a different set of hyperparameters using "grid search".

        Args:
            pipe_components: list of dicts; config if components

        Returns:
            grid search iterator
        """
        list_of_variants = []
        # find in config keys for grid search
        for i, component in enumerate(pipe_components):
            if component is not None:
                for key, item in component.items():
                    if isinstance(item, dict) and 'grid_search' in item:
                        var_list = [{var: [i, key]} for var in item['grid_search']]
                        list_of_variants.append(var_list)
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
    def change_load_path(config: List[dict],
                         pipe_ind: int,
                         save_path: Union[str, Path]) -> List[dict]:
        """
        Change save_path and load_path attributes in standard DeepPavlov config.

        Args:
            config: dict; the chainer content.
            pipe_ind: int; pipeline number
            save_path: str; path to root folder where will be saved all checkpoints

        Returns:
            config: dict; new config with changed save and load paths
        """
        base_path = Path(save_path).joinpath(f'pipe_{pipe_ind}')
        for component in config:
            if 'save_path' in component:
                if len(Path(component['save_path']).name.split('.')) != 1:
                    component['save_path'] = str(base_path / Path(component['save_path']).name)
                else:
                    component['save_path'] = str(base_path)
            if 'load_path' in component:
                if len(Path(component['load_path']).name.split('.')) != 1:
                    component['load_path'] = str(base_path / Path(component['load_path']).name)
                else:
                    component['load_path'] = str(base_path)
        return config
