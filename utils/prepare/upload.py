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

    print(config_in_file)
    config_in = parse_config(config_in_file)
    config_in_file = find_config(config_in_file)

    model_path = Path(config_in['metadata']['variables']['MODEL_PATH']).expanduser()
    models_path = Path(config_in['metadata']['variables']['MODELS_PATH']).expanduser()
    model_name, class_name = config_in_file.stem, config_in_file.parent.name
    
    if str(model_name) not in str(model_path):
        raise(f'{model_name} is not the path of the {model_path}')
    
    arcname = str(model_path).split("models/")[1]
    tar_path = models_path/model_name
    tmp_folder = f'/tmp/'
    tmp_tar = tmp_folder + f'{model_name}.tar.gz'

    print("model_path", model_path)
    print("class_name", class_name)
    print("model_name", model_name)
    
    print("Start tarring")
    archive = tarfile.open(tmp_tar, "w|gz")
    archive.add(model_path, arcname=arcname)
    archive.close()
    print("Stop tarring")

    print("Calculating hash")
    main(tmp_tar)

    print("tmp_tar", tmp_tar)
    command = f'scp -r {tmp_folder}{model_name}* share.ipavlov.mipt.ru:/home/export/v1/{class_name}'
    donwload_url = f'http://files.deeppavlov.ai/v1/{class_name}/{model_name}.tar.gz'
    print(command, donwload_url, sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_in", help="path to a config", type=str)
    args = parser.parse_args()
    upload(args.config_in)
