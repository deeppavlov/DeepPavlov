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

import sqlite3
from typing import List, Any, Dict, Optional
from random import Random
from pathlib import Path
import os

from overrides import overrides

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.file import read_json
from deeppavlov.core.data.utils import download
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.infer import build_model_from_config

logger = get_logger(__name__)


@register('squad_wrapper')
class SquadWrapper(Component):
    """
    Load a SQLite database, read data batches and get docs content.
    """

    def __init__(self, config_path1, *args, **kwargs):
        self.squad_pipeline = build_model_from_config(read_json(os.path.join(os.path.dirname(__file__), "../../configs/squad/squad_ru.json")))

    def __call__(self, question, contexts, *args, **kwargs):
        answers = []
        for c in contexts[0][0]:
            squad_answer = self.squad_pipeline([(c, question[0])])
            answers.append(squad_answer[0][0])
        return answers
