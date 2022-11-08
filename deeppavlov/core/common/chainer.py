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

import pickle
from itertools import islice
from logging import getLogger
from types import FunctionType
from typing import Union, Tuple, List, Optional, Hashable, Reversible

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.models.serializable import Serializable

log = getLogger(__name__)


class Chainer(Component):
    """
    Builds a component pipeline from heterogeneous components (Rule-based/ML/DL). It allows to train
    and infer models in a pipeline as a whole.

    Attributes:
        pipe: list of components and their input and output variable names for inference
        train_pipe: list of components and their input and output variable names for training and evaluation
        in_x: names of inputs for pipeline inference mode
        out_params: names of pipeline inference outputs
        in_y: names of additional inputs for pipeline training and evaluation modes
        forward_map: list of all variables in chainer's memory after  running every component in ``self.pipe``
        train_map: list of all variables in chainer's memory after  running every component in ``train_pipe.pipe``
        main: reference to the main component

    Args:
        in_x: names of inputs for pipeline inference mode
        out_params: names of pipeline inference outputs
        in_y: names of additional inputs for pipeline training and evaluation modes
    """

    def __init__(self, in_x: Union[str, list] = None, out_params: Union[str, list] = None,
                 in_y: Union[str, list] = None, *args, **kwargs) -> None:
        self.pipe: List[Tuple[Tuple[List[str], List[str]], List[str], Component]] = []
        self.train_pipe = []
        if isinstance(in_x, str):
            in_x = [in_x]
        if isinstance(in_y, str):
            in_y = [in_y]
        if isinstance(out_params, str):
            out_params = [out_params]
        self.in_x = in_x or ['x']
        self.in_y = in_y or ['y']
        self.out_params = out_params or self.in_x

        self.forward_map = set(self.in_x)
        self.train_map = self.forward_map.union(self.in_y)

        self._components_dict = {}

        self.main = None

    def __getitem__(self, item):
        if isinstance(item, int):
            in_params, out_params, component = self.train_pipe[item]
            return component
        return self._components_dict[item]

    def _ipython_key_completions_(self):
        return self._components_dict.keys()

    def __repr__(self):
        reversed_components_dict = {v: f'{repr(k)}: ' for k, v in self._components_dict.items()
                                    if isinstance(v, Hashable)}

        components_list = []
        for in_params, out_params, component in self.train_pipe:
            component_repr = repr(component)
            if isinstance(component, Hashable) and component in reversed_components_dict:
                component_repr = reversed_components_dict[component] + component_repr
            else:
                for k, v in self._components_dict.items():
                    if v is component:
                        component_repr = f'{k}: {component_repr}'
                        break
            components_list.append(component_repr)

        return f'Chainer[{", ".join(components_list)}]'

    def _repr_pretty_(self, p, cycle):
        """method that defines ``Struct``'s pretty printing rules for iPython

        Args:
            p (IPython.lib.pretty.RepresentationPrinter): pretty printer object
            cycle (bool): is ``True`` if pretty detected a cycle
        """
        if cycle:
            p.text('Chainer(...)')
        else:
            with p.group(8, 'Chainer[', ']'):
                reversed_components_dict = {v: k for k, v in self._components_dict.items()
                                            if isinstance(v, Hashable)}
                # p.pretty(self.__prepare_repr())
                for i, (in_params, out_params, component) in enumerate(self.train_pipe):
                    if i > 0:
                        p.text(',')
                        p.breakable()
                    if isinstance(component, Hashable) and component in reversed_components_dict:
                        p.pretty(reversed_components_dict[component])
                        p.text(': ')
                    else:
                        for k, v in self._components_dict.items():
                            if v is component:
                                p.pretty(k)
                                p.text(': ')
                                break
                    p.pretty(component)

    def append(self, component: Union[Component, FunctionType], in_x: [str, list, dict] = None,
               out_params: [str, list] = None, in_y: [str, list, dict] = None, main: bool = False):
        if isinstance(in_x, str):
            in_x = [in_x]
        if isinstance(in_y, str):
            in_y = [in_y]
        if isinstance(out_params, str):
            out_params = [out_params]
        in_x = in_x or self.in_x

        if isinstance(in_x, dict):
            x_keys, in_x = zip(*in_x.items())
        else:
            x_keys = []
        out_params = out_params or in_x
        if in_y is not None:
            if isinstance(in_y, dict):
                y_keys, in_y = zip(*in_y.items())
            else:
                y_keys = []
            keys = x_keys + y_keys

            if bool(x_keys) != bool(y_keys):
                raise ConfigError('`in` and `in_y` for a component have to both be lists or dicts')

            component: NNModel
            main = True
            assert self.train_map.issuperset(in_x + in_y), ('Arguments {} are expected but only {} are set'
                                                            .format(in_x + in_y, self.train_map))
            preprocessor = Chainer(self.in_x, in_x + in_y, self.in_y)
            for (t_in_x_keys, t_in_x), t_out, t_component in self.train_pipe:
                if t_in_x_keys:
                    t_in_x = dict(zip(t_in_x_keys, t_in_x))
                preprocessor.append(t_component, t_in_x, t_out)

            def train_on_batch(*args, **kwargs):
                preprocessed = preprocessor.compute(*args, **kwargs)
                if len(in_x + in_y) == 1:
                    preprocessed = [preprocessed]
                if keys:
                    return component.train_on_batch(**dict(zip(keys, preprocessed)))
                else:
                    return component.train_on_batch(*preprocessed)

            self.train_on_batch = train_on_batch
            self.process_event = component.process_event
        if main:
            self.main = component
        if self.forward_map.issuperset(in_x):
            self.pipe.append(((x_keys, in_x), out_params, component))
            self.forward_map = self.forward_map.union(out_params)

        if self.train_map.issuperset(in_x):
            self.train_pipe.append(((x_keys, in_x), out_params, component))
            self.train_map = self.train_map.union(out_params)
        else:
            raise ConfigError('Arguments {} are expected but only {} are set'.format(in_x, self.train_map))

    def compute(self, x, y=None, targets=None):
        if targets is None:
            targets = self.out_params
        in_params = list(self.in_x)
        if len(in_params) == 1:
            args = [x]
        else:
            args = list(zip(*x))

        if y is None:
            pipe = self.pipe
        else:
            pipe = self.train_pipe
            if len(self.in_y) == 1:
                args.append(y)
            else:
                args += list(zip(*y))
            in_params += self.in_y

        return self._compute(*args, pipe=pipe, param_names=in_params, targets=targets)

    def __call__(self, *args):
        return self._compute(*args, param_names=self.in_x, pipe=self.pipe, targets=self.out_params)

    @staticmethod
    def _compute(*args, param_names, pipe, targets):
        expected = set(targets)
        final_pipe = []
        for (in_keys, in_params), out_params, component in reversed(pipe):
            if expected.intersection(out_params):
                expected = expected - set(out_params) | set(in_params)
                final_pipe.append(((in_keys, in_params), out_params, component))
        final_pipe.reverse()
        if not expected.issubset(param_names):
            raise RuntimeError(f'{expected} are required to compute {targets} but were not found in memory or inputs')
        pipe = final_pipe

        mem = dict(zip(param_names, args))
        del args

        for (in_keys, in_params), out_params, component in pipe:
            x = [mem[k] for k in in_params]
            if in_keys:
                res = component.__call__(**dict(zip(in_keys, x)))
            else:
                res = component.__call__(*x)
            if len(out_params) == 1:
                mem[out_params[0]] = res
            else:
                mem.update(zip(out_params, res))

        res = [mem[k] for k in targets]
        if len(res) == 1:
            res = res[0]
        return res

    def batched_call(self, *args: Reversible, batch_size: int = 16) -> Union[list, Tuple[list, ...]]:
        """
        Partitions data into mini-batches and applies :meth:`__call__` to each batch.

        Args:
            args: input data, each element of the data corresponds to a single model inputs sequence.
            batch_size: the size of a batch.

        Returns:
            the model output as if the data was passed to the :meth:`__call__` method.
        """
        args = [iter(arg) for arg in args]
        answer = [[] for _ in self.out_params]

        while True:
            batch = [list(islice(arg, batch_size)) for arg in args]
            if not any(batch):  # empty batch, reached the end
                break

            curr_answer = self.__call__(*batch)
            if len(self.out_params) == 1:
                curr_answer = [curr_answer]

            for y, curr_y in zip(answer, curr_answer):
                y.extend(curr_y)

        if len(self.out_params) == 1:
            answer = answer[0]
        return answer

    def get_main_component(self) -> Optional[Serializable]:
        try:
            return self.main or self.pipe[-1][-1]
        except IndexError:
            log.warning('Cannot get a main component for an empty chainer')
            return None

    def save(self) -> None:
        main_component = self.get_main_component()
        if isinstance(main_component, Serializable):
            main_component.save()

    def load(self) -> None:
        for in_params, out_params, component in self.train_pipe:
            if callable(getattr(component, 'load', None)):
                component.load()

    def reset(self) -> None:
        for in_params, out_params, component in self.train_pipe:
            if callable(getattr(component, 'reset', None)):
                component.reset()

    def destroy(self):
        if hasattr(self, 'train_pipe'):
            for in_params, out_params, component in self.train_pipe:
                if callable(getattr(component, 'destroy', None)):
                    component.destroy()
            self.train_pipe.clear()
        if hasattr(self, 'pipe'):
            self.pipe.clear()
        super().destroy()
