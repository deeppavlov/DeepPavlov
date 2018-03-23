"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.component import Component


class Chainer(Component):
    def __init__(self, in_x, out_params, in_y=None, *args, **kwargs):
        self.pipe = []
        self.train_pipe = []
        self.in_x = in_x
        self.in_y = in_y or []
        self.out_params = out_params

        self.forward_map = set(self.in_x)
        self.train_map = self.forward_map.union(self.in_y)

        self.main = None

    def append(self, in_x, out_params, component, in_y=None, main=False):
        if in_y is not None:
            main = True
            assert self.train_map.issuperset(in_x+in_y), ('Arguments {} are expected but only {} are set'
                                                          .format(in_x+in_y, self.train_map))
            preprocessor = Chainer(self.in_x, in_x+in_y, self.in_y)
            for t_in_x, t_out, t_component in self.train_pipe:
                preprocessor.append(t_in_x, t_out, t_component)

            def train_on_batch(*args, **kwargs):
                preprocessed = zip(*preprocessor(*args, **kwargs))
                return component.train_on_batch(*preprocessed)

            self.train_on_batch = train_on_batch
        if main:
            self.main = component
        if self.forward_map.issuperset(in_x):
            self.pipe.append((in_x, out_params, component))
            self.forward_map = self.forward_map.union(out_params)
        if self.train_map.issuperset(in_x):
            self.train_pipe.append((in_x, out_params, component))
            self.train_map = self.train_map.union(out_params)
        else:
            raise ConfigError('Arguments {} are expected but only {} are set'.format(in_x, self.train_map))

    def __call__(self, x, y=None, to_return=None):
        in_params = list(self.in_x)
        mem = {}

        if to_return is None:
            to_return = self.out_params
        if self.forward_map.issuperset(to_return):
            pipe = self.pipe
        elif y is None:
            raise RuntimeError('Expected to return {} but only {} are set in memory'
                               .format(to_return, self.forward_map))
        elif self.train_map.issuperset(to_return):
            in_params += self.in_y
            pipe = self.train_pipe

            if len(self.in_y) == 1:
                y = [y]
            else:
                y = list(zip(*y))
            for i, k in enumerate(self.in_y):
                mem[k] = y[i]
        else:
            raise RuntimeError('Expected to return {} but only {} are set in memory'
                               .format(to_return, self.train_map))
        if len(self.in_x) == 1:
            x = [x]
        else:
            x = list(zip(*x))
        for i, k in enumerate(self.in_x):
            mem[k] = x[i]

        del x, y

        for in_params, out_params, component in pipe:
            res = component(*[mem[k] for k in in_params])
            if len(out_params) == 1:
                mem[out_params[0]] = res
            else:
                for k, v in zip(out_params, res):
                    mem[k] = v

        res = [mem[k] for k in to_return]
        if len(res) == 1:
            return res[0]
        return list(zip(*res))

    def get_main_component(self):
        return self.main or self.pipe[-1][-1]

    def save(self):
        self.get_main_component().save()
