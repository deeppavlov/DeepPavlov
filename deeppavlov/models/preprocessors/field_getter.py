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

from typing import List, Dict, Union, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('field_getter')
class FieldGetter(Component):
    def __init__(self, field, *args, **kwargs):
        self.field = field

    def __call__(self, batch: Union[List[Dict], Tuple[Dict]], **kwargs):
        return [self(item) for item in batch]
