# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

from pathlib import Path

import deeppavlov

from deeppavlov import build_model
from deeppavlov.core.common.file import read_json
from deeppavlov.utils.pip_wrapper import install_from_config


vectorizer = None


def get_vectorizer():
    global vectorizer
    if vectorizer:
        return vectorizer
    else:
        model_config = read_json(Path(deeppavlov.__path__[0]) / "configs/vectorizer/fasttext_vectorizer.json")
        install_from_config(model_config)
        vectorizer = build_model(model_config)
        return vectorizer
