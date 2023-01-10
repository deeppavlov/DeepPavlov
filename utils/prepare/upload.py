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

import argparse
import pathlib
import tarfile
from pathlib import Path

from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.common.file import find_config
from hashes import main


def upload(config_in_file: str, tar_name: str, tar_output_dir: Path):
    if not tar_output_dir.exists():
        raise RuntimeError(f'A folder {tar_output_dir} does not exist')

    print(f'Config: {config_in_file}')
    if not Path(config_in_file).exists():
        raise RuntimeError(f'A config {config_in_file} does not exist')

    config_in = parse_config(config_in_file)
    config_in_file = find_config(config_in_file)

    model_path = Path(config_in['metadata']['variables']['MODEL_PATH']).expanduser()
    model_name, class_name = config_in_file.stem, config_in_file.parent.name

    if tar_name is None:
        tar_name = f'{model_name}'
        print(f'tar_name set to {tar_name}')

    full_tar_name = tar_output_dir / f'{tar_name}.tar.gz'
    if Path(full_tar_name).exists():
        raise RuntimeError(f'An archive {Path(full_tar_name)} already exists')

    print(f'model_path: {model_path}')
    print(f'class_name: {class_name}')
    print(f'model_name: {model_name}')
    print(f'Start tarring to {full_tar_name}')
    with tarfile.open(str(full_tar_name), "w|gz") as archive:
        archive.add(model_path, arcname=pathlib.os.sep)

    print("Stop tarring")
    print(f'Tar archive: {Path(full_tar_name)} has been created')

    print("Calculating hash")
    main(full_tar_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_in', help='path to a config', type=str)
    parser.add_argument('-n', '--tar_name', help='name of the tar archive (without tar.gz extension)',
                        default=None, required=False, type=str)
    parser.add_argument('-o', '--tar_output_dir', help='dir to save a tar archive', default='./',
                        required=False, type=Path)
    args = parser.parse_args()
    upload(args.config_in, args.tar_name, args.tar_output_dir)
