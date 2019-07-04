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

from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress


@register('kbqa_reader')
class KBQAReader(DatasetReader):
    """Class to read test set of questions and answers for knowledge base question answering"""

    def read(self, data_path: str):
        data_path = Path(data_path)
        files = list(data_path.glob('*.txt'))
        test_set_filename = "test_set_with_answers.txt"
        if test_set_filename not in {file_path.name for file_path in files}:
            url = 'http://files.deeppavlov.ai/kbqa/test_set_with_answers.zip'
            data_path.mkdir(exist_ok=True, parents=True)
            download_decompress(url, data_path)
        dataset = {}

        dataset["test"] = self.parse_ner_file(data_path / test_set_filename)
        dataset["train"] = []
        dataset["valid"] = []
        return dataset

    def parse_ner_file(self, file_name: Path):
        samples = []
        with file_name.open(encoding='utf8') as f:
            for line in f:
                line_split = line.strip('\n').split('\t')
                samples.append((line_split[0], tuple(line_split[1:])))

        return samples
