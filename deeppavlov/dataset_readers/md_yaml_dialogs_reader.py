# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from logging import getLogger
from overrides import overrides
from pathlib import Path
from typing import Dict

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.dataset_readers.dto.rasa.domain_knowledge import DomainKnowledge
from deeppavlov.dataset_readers.dto.rasa.nlu import Intents
from deeppavlov.dataset_readers.dto.rasa.stories import Stories

log = getLogger(__name__)


class RASADict(dict):
    def __add__(self, oth):
        return RASADict()

@register('md_yaml_dialogs_reader')
class MD_YAML_DialogsDatasetReader(DatasetReader):
    """
    Reads dialogs from dataset composed of ``stories.md``, ``nlu.md``, ``domain.yml`` .

    ``stories.md`` is to provide the dialogues dataset for model to train on. The dialogues
    are represented as user messages labels and system response messages labels: (not texts, just action labels).
    This is so to distinguish the NLU-NLG tasks from the actual dialogues storytelling experience: one
    should be able to describe just the scripts of dialogues to the system.

    ``nlu.md`` is contrariwise to provide the NLU training set irrespective of the dialogues scripts.

    ``domain.yml`` is to desribe the task-specific domain and serves two purposes:
    provide the NLG templates and provide some specific configuration of the NLU
    """

    _USER_SPEAKER_ID = 1
    _SYSTEM_SPEAKER_ID = 2

    VALID_DATATYPES = ('trn', 'val', 'tst')

    NLU_FNAME = "nlu.md"
    DOMAIN_FNAME = "domain.yml"

    @classmethod
    def _data_fname(cls, datatype: str) -> str:
        assert datatype in cls.VALID_DATATYPES, f"wrong datatype name: {datatype}"
        return f"stories-{datatype}.md"

    @classmethod
    @overrides
    def read(cls, data_path: str, fmt = "md") -> Dict[str, Dict]:
        """
        Parameters:
            data_path: path to read dataset from

        Returns:
            dictionary that contains
            ``'train'`` field with dialogs from ``'stories-trn.md'``,
            ``'valid'`` field with dialogs from ``'stories-val.md'`` and
            ``'test'`` field with dialogs from ``'stories-tst.md'``.
            Each field is a list of tuples ``(x_i, y_i)``.
        """
        domain_fname = cls.DOMAIN_FNAME
        nlu_fname = cls.NLU_FNAME if fmt in ("md", "markdown") else cls.NLU_FNAME.replace('.md', f'.{fmt}')
        stories_fnames = tuple(cls._data_fname(dt) for dt in cls.VALID_DATATYPES)
        required_fnames = stories_fnames + (nlu_fname, domain_fname)
        for required_fname in required_fnames:
            required_path = Path(data_path, required_fname)
            if not required_path.exists():
                log.error(f"INSIDE MLU_MD_DialogsDatasetReader.read(): "
                          f"{required_fname} not found with path {required_path}")

        domain_path = Path(data_path, domain_fname)
        domain_knowledge = DomainKnowledge.from_yaml(domain_path)
        nlu_fpath = Path(data_path, nlu_fname)
        intents = Intents.from_file(nlu_fpath)

        short2long_subsample_name = {"trn": "train",
                                     "val": "valid",
                                     "tst": "test"}

        data = dict()
        for subsample_name_short in cls.VALID_DATATYPES:
            story_fpath = Path(data_path, cls._data_fname(subsample_name_short))
            with open(story_fpath) as f:
                story_lines = f.read().splitlines()
            stories = Stories.from_stories_lines_md(story_lines)

            data[short2long_subsample_name[subsample_name_short]] = RASADict({
                            "story_lines": stories,
                            "domain": domain_knowledge,
                            "nlu_lines": intents})
        data = RASADict(data)
        return data