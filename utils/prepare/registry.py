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

import json
import pkgutil
from importlib import import_module, reload

import deeppavlov
from deeppavlov.core.common.metrics_registry import _registry_path as m_registry_path, _REGISTRY as M_REGISTRY
from deeppavlov.core.common.registry import _registry_path as c_registry_path, _REGISTRY as C_REGISTRY

if __name__ == '__main__':
    C_REGISTRY.clear()
    M_REGISTRY.clear()

    for _, pkg_name, _ in pkgutil.walk_packages(deeppavlov.__path__, deeppavlov.__name__ + '.'):
        if pkg_name not in ('deeppavlov.core.common.registry', 'deeppavlov.core.common.metrics_registry'):
            reload(import_module(pkg_name))

    with c_registry_path.open('w', encoding='utf-8') as f:
        json.dump(dict(sorted(C_REGISTRY.items())), f, indent=2)

    with m_registry_path.open('w', encoding='utf-8') as f:
        json.dump(dict(sorted(M_REGISTRY.items())), f, indent=2)
