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
import os
import shutil
import tarfile
from pathlib import Path

from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.common.file import find_config
from hashes import main


def upload(config_in_file):
    config_in = parse_config(config_in_file)
    config_in_file = find_config(config_in_file)

    model_path = Path(config_in['metadata']['variables']['MODEL_PATH']).expanduser()

    model_name, class_name = config_in_file.stem, config_in_file.parent.name

    tmp_dir = f'/tmp/{class_name}'
    tmp_tar = f'/tmp/{class_name}/{model_name}.tar.gz'
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.mkdir(tmp_dir)

    with tarfile.open(tmp_tar, "w:gz") as tar:
        tar.add(model_path, arcname=model_name)

    main(tmp_tar)

    command = f'scp -r {tmp_dir} share.ipavlov.mipt.ru:/home/export/v1/'
    donwload_url = f'http://files.deeppavlov.ai/v1/{class_name}/{model_name}.tar.gz'
    print(command, donwload_url, sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_in", help="path to a config", type=str)
    args = parser.parse_args()
    upload(args.config_in)
