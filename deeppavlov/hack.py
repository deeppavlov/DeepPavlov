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

from pathlib import Path
import sys

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.commands.train import build_model_from_config
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json

log = get_logger(__name__)

CONFIG_PATH = 'configs/odqa/ru_odqa_infer_prod.json'
chainer = build_model_from_config(read_json(CONFIG_PATH))

def main():

    while True:
        query = input("Question: ")
        output = chainer([query])
        print(*output)

if __name__ == "__main__":
    main()
