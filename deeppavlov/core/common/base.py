# Copyright 2021 Neural Networks and Deep Learning lab, MIPT
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

from types import FunctionType
from typing import List, Optional, Union

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.models.component import Component


class Element:
    """DeepPavlov model pipeline element."""
    def __init__(self, component: Union[Component, FunctionType],
                 x: Optional[Union[str, list]] = None,
                 out: Optional[Union[str, list]] = None,
                 y: Optional[Union[str, list]] = None,
                 main: bool = False) -> None:
        """
        Args:
            component: Pipeline component object.
            x: Names of the component inference inputs. Output from other pipeline elements with such names will be fed
                to the input of this component.
            out: Names of the component inference outputs. Component outputs can be fed to other pipeline elements
                using this names.
            y: Names of additional inputs (targets) for component training and evaluation.
            main: Set True if this is the main component. Main component is trained during model training process.
        """
        self.component = component
        self.x = x
        self.y = y
        self.out = out
        self.main = main


class Model(Chainer):
    """Builds a component pipeline to train and infer models."""
    def __init__(self, x: Optional[Union[str, list]] = None,
                 out: Optional[Union[str, list]] = None,
                 y: Optional[Union[str, list]] = None,
                 pipe: Optional[List[Element]] = None) -> None:
        """
        Args:
            x: Names of pipeline inference inputs.
            out: Names of pipeline inference outputs.
            y: Names of additional inputs (targets) for pipeline training and evaluation.
            pipe: List of pipeline elements.
        """
        super().__init__(in_x=x, out_params=out, in_y=y)
        if pipe is not None:
            for element in pipe:
                self.append(element.component, element.x, element.out, element.y, element.main)
