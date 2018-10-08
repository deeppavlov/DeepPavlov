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
from typing import Union

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.nn_model import NNModel


class Chainer(Component):
    """
    Builds an agent/component pipeline from heterogeneous components (Rule-based/ML/DL). It allows to train
    and infer models in a pipeline as a whole.
    """
    def __init__(self, in_x: Union[str, list] = None, out_params: Union[str, list] = None,
                 in_y: Union[str, list] = None, *args, as_component: bool = False, **kwargs):
        self.pipe = []
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

        self.main = None

        if as_component:
            self._predict = self._predict_as_component

    def append(self, component: Component, in_x: [str, list, dict]=None, out_params: [str, list]=None,
               in_y: [str, list, dict]=None, main=False):
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
            assert self.train_map.issuperset(in_x+in_y), ('Arguments {} are expected but only {} are set'
                                                          .format(in_x+in_y, self.train_map))
            preprocessor = Chainer(self.in_x, in_x+in_y, self.in_y)
            for (t_in_x_keys, t_in_x), t_out, t_component in self.train_pipe:
                if t_in_x_keys:
                    t_in_x = dict(zip(t_in_x_keys, t_in_x))
                preprocessor.append(t_component, t_in_x, t_out)

            def train_on_batch(*args, learning_rate=None, **kwargs):
                preprocessed = zip(*preprocessor(*args, **kwargs))
                args, kwargs = [], {}
                if keys:
                    kwargs = dict(zip(keys, preprocessed))
                else:
                    args = preprocessed
                if learning_rate is not None:
                    kwargs['learning_rate'] = learning_rate
                return component.train_on_batch(*args, **kwargs)

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

    def __call__(self, *args, **kwargs):
        return self._predict(*args, **kwargs)

    def _predict(self, x, y=None, to_return=None):
        in_params = list(self.in_x)
        if len(in_params) == 1:
            args = [x]
        else:
            args = list(zip(*x))

        if to_return is None:
            to_return = self.out_params
        if self.forward_map.issuperset(to_return):
            pipe = self.pipe
        elif y is None:
            raise RuntimeError('Expected to return {} but only {} are set in memory'
                               .format(to_return, self.forward_map))
        elif self.train_map.issuperset(to_return):
            pipe = self.train_pipe
            if len(self.in_y) == 1:
                args.append(y)
            else:
                args += list(zip(*y))
            in_params += self.in_y
        else:
            raise RuntimeError('Expected to return {} but only {} are set in memory'
                               .format(to_return, self.train_map))

        mem = dict(zip(in_params, args))
        del args, x, y

        for (in_keys, in_params), out_params, component in pipe:
            x = [mem[k] for k in in_params]
            if in_keys:
                res = component(**dict(zip(in_keys, x)))
            else:
                res = component(*x)
            if len(out_params) == 1:
                mem[out_params[0]] = res
            else:
                mem.update(zip(out_params, res))

        res = [mem[k] for k in to_return]
        if len(res) == 1:
            return res[0]
        return list(zip(*res))

    def _predict_as_component(self, *args):
        mem = dict(zip(self.in_x, args))

        for (in_keys, in_params), out_params, component in self.pipe:
            x = [mem[k] for k in in_params]
            if in_keys:
                res = component(**dict(zip(in_keys, x)))
            else:
                res = component(*x)

            if len(out_params) == 1:
                mem[out_params[0]] = res
            else:
                mem.update(zip(out_params, res))

        res = [mem[k] for k in self.out_params]
        if len(res) == 1:
            res = res[0]
        return res

    def get_main_component(self):
        return self.main or self.pipe[-1][-1]

    def save(self):
        self.get_main_component().save()

    def load(self):
        for in_params, out_params, component in self.pipe:
            if inspect.ismethod(getattr(component, 'load', None)):
                component.load()

    def reset(self):
        for in_params, out_params, component in self.pipe:
            if inspect.ismethod(getattr(component, 'reset', None)):
                component.reset()

    def destroy(self):
        for in_params, out_params, component in self.pipe:
            if inspect.ismethod(getattr(component, 'destroy', None)):
                component.destroy()
