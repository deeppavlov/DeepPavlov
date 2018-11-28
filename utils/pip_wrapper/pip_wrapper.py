import re
import subprocess
import sys
from pathlib import Path
import os

from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

_tf_re = re.compile(r'\s*tensorflow\s*([<=>;]|$)')


def install(*packages):
    if any(_tf_re.match(package) for package in packages)\
            and b'tensorflow-gpu' in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'],
                                                             env=os.environ.copy()):
        log.warn('found tensorflow-gpu installed, so upgrading it instead of tensorflow')
        packages = [_tf_re.sub(r'tensorflow-gpu\1', package) for package in packages]
    result = subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                   *[re.sub(r'\s', '', package) for package in packages]],
                                   env=os.environ.copy())
    return result


def install_from_config(config: [str, Path, dict]):
    config = parse_config(config)
    requirements_files = config.get('metadata', {}).get('requirements', [])

    if not requirements_files:
        log.warn('No requirements found in config')
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
