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
import csv

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.commands.train import build_model_from_config
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.utils import expand_path, get_project_root

log = get_logger(__name__)

CONFIG_PATH = str(get_project_root()) + '/deeppavlov/configs/odqa/odqa_hack.json'
print(CONFIG_PATH)
chainer = build_model_from_config(read_json(CONFIG_PATH))


def main():
    with open('conversation_result.csv', 'w') as csvfile:
        while True:
            writer = csv.writer(csvfile, delimiter=';')
            try:
                query = input("Question: ")
                answers = chainer([query.strip()])
                for answer in answers:
                    if '\n' in answer:
                        answer = answer.split('\n')[0]
                    print(answer)
                writer.writerow([query, *answers])
            except Exception:
                answer = "Я не знаю ответ."
                try:
                    writer.writerow([query, [answer]*3])
                except Exception:
                    writer.writerow(["Неизвестный вопрос.", [answer]*3])
                print(answer * 3)


if __name__ == "__main__":
    main()
