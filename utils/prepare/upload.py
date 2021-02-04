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
import json
import shutil
import tarfile
import argparse
from hashes import main
from logging import getLogger

logger = getLogger(__name__)

def upload(config_in_file):
    
    with open(config_in_file, 'r') as f:
        config_in = json.load(f)
    
    model_path = config_in['metadata']['variables']['MODEL_PATH']
    
    if 'TRANSFORMER' in config_in['metadata']['variables']:
        transformers = config_in['metadata']['variables']['TRANSFORMER']
        model_path = model_path.replace('{TRANSFORMER}', transformers)
        
    model_path = model_path.replace('{MODELS_PATH}', '~/.deeppavlov/models')
    model_name = os.path.splitext(os.path.basename(config_in_file))[0]
    class_name = os.path.basename(os.path.dirname(config_in_file))

    shutil.rmtree("/tmp/"+class_name, ignore_errors=True)
    os.mkdir("/tmp/"+class_name)

    with tarfile.open("/tmp/"+class_name+"/"+model_name+".tar.gz", "w:gz") as tar:
        tar.add(os.path.expanduser(model_path), arcname=model_name)
    
    files = ["/tmp/"+class_name+"/"+model_name+".tar.gz", "/tmp/"+class_name+"/"+model_name+".md5"]
    main(files[0], files[1])
    
    command="scp -r /tmp/" + class_name + " share.ipavlov.mipt.ru:/home/export/v1/"
    print(command)
    
    donwload_url = "http://files.deeppavlov.ai/v1/"+class_name+"/"+model_name+".tar.gz"
    print(donwload_url)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_in", help="path to a config", type=str)
    args = parser.parse_args()
    upload(args.config_in)