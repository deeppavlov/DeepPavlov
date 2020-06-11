# Copyright 2019 Alexey Romanov
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

import json
from logging import getLogger
from pathlib import Path
from typing import List, Dict, Any, Optional

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done

log = getLogger(__name__)


@register('snips_reader')
class SnipsReader(DatasetReader):
    """The class to download and read Snips NLU Benchmark dataset (custom intents section).

    See https://github.com/snipsco/nlu-benchmark.
    """

    # noinspection PyAttributeOutsideInit
    @overrides
    def read(self, data_path: str, queries_per_intent: Optional[int] = None, test_validate_split: float = 0.5,
             *args, **kwargs) -> \
            Dict[str, List[Dict[str, Any]]]:
        """
        Each query in the output has the following form:
            { 'intent': intent_name,
              'data': [ { 'text': text, ('entity': slot_name)? } ]
            }

        Args:
            data_path: A path to a folder with dataset files.
            queries_per_intent: Number of queries to load for each intent. None to load all.
                If the requested number is greater than available in file, all queries are returned.
            test_validate_split: Proportion of `_validate` files to be used as test dataset (since Snips
                is split into training and validation sets without a separate test set).
        """
        data_path = Path(data_path)
        intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                   'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

        if not is_done(data_path):
            url = 'http://files.deeppavlov.ai/datasets/snips.tar.gz'
            log.info('[downloading data from {} to {}]'.format(url, data_path))
            download_decompress(url, data_path)
            mark_done(data_path)

        use_full_file = queries_per_intent is None or queries_per_intent > 70
        training_data = []
        validation_data = []
        test_data = []

        for intent in intents:
            intent_path = data_path / intent
            train_file_name = f"train_{intent}{'_full' if use_full_file else ''}.json"
            validate_file_name = f"validate_{intent}.json"

            train_queries = self._load_file(intent_path / train_file_name, intent, queries_per_intent)
            validate_queries = self._load_file(intent_path / validate_file_name, intent, queries_per_intent)
            num_test_queries = round(len(validate_queries) * test_validate_split)

            training_data.extend(train_queries)
            validation_data.extend(validate_queries[num_test_queries:])
            test_data.extend(validate_queries[:num_test_queries])

        return {'train': training_data, 'valid': validation_data, 'test': test_data}

    @staticmethod
    def _load_file(path: Path, intent: str, num_queries: Optional[int]):
        with path.open(encoding='latin_1') as f:
            data = json.load(f)

        # restrict number of queries
        queries = data[intent][:num_queries]
        for query in queries:
            query['intent'] = intent
        return queries
