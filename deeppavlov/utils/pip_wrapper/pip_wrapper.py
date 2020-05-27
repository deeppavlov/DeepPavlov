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

import os
import re
import subprocess
import sys
from logging import getLogger
from pathlib import Path

from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.data.utils import get_all_elems_from_json

log = getLogger(__name__)

_tf_re = re.compile(r'\s*tensorflow\s*([<=>;]|$)')


def install(*packages):
    if any(_tf_re.match(package) for package in packages) \
            and b'tensorflow-gpu' in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'],
                                                             env=os.environ.copy()):
        log.warning('found tensorflow-gpu installed, so upgrading it instead of tensorflow')
        packages = [_tf_re.sub(r'tensorflow-gpu\1', package) for package in packages]
    result = subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                    *[re.sub(r'\s', '', package) for package in packages]],
                                   env=os.environ.copy())
    return result


def get_config_requirements(config: [str, Path, dict]):
    config = parse_config(config)

    requirements = set()
    for req in config.get('metadata', {}).get('requirements', []):
        requirements.add(req)

    config_references = [expand_path(config_ref) for config_ref in get_all_elems_from_json(config, 'config_path')]
    requirements |= {req for config in config_references for req in get_config_requirements(config)}

    return requirements


def install_from_config(config: [str, Path, dict]):
    requirements_files = get_config_requirements(config)

    if not requirements_files:
        log.warning('No requirements found in config')
        return

    requirements = []
    for rf in requirements_files:
        with expand_path(rf).open(encoding='utf8') as f:
            for line in f:
                line = re.sub(r'\s', '', line.strip())
                if line and not line.startswith('#') and line not in requirements:
                    requirements.append(line)

    for r in requirements:
        install(r)
