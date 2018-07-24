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

import json
import pickle


def read_json(fpath):
    with open(fpath, encoding='utf8') as fin:
        return json.load(fin)


def save_json(data, fpath):
    with open(fpath, 'w', encoding='utf8') as fout:
        return json.dump(data, fout, ensure_ascii=False, indent=2)


def save_pickle(data, fpath):
    with open(fpath, 'wb') as fout:
        pickle.dump(data, fout)


def load_pickle(fpath):
    with open(fpath, 'rb') as fin:
        return pickle.load(fin)
