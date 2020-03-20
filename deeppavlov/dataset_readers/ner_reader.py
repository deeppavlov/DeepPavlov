import os
from logging import getLogger
from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress

log = getLogger(__name__)


@register('ner_reader')
class NerReader(DatasetReader):
    """
    Class to read training datasets in the CoNLL2003 format; support both IOB and IOBES tagging schemes.
    """

    def read(self,
             data_path: str,
             dataset_name: str = None,
             provide_pos: bool = False,
             provide_chunk: bool = False,
             tagging_scheme: str = "iob"):

        self.provide_pos = provide_pos
        self.provide_chunk = provide_chunk
        self.tagging_scheme = tagging_scheme
        data_path = Path(data_path)
        files = list(data_path.glob('*.txt'))
        if 'train.txt' not in {file_path.name for file_path in files}:
            if dataset_name == 'conll2003':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/conll2003_v2.tar.gz'
            elif dataset_name == 'collection_rus':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/collection3_v2.tar.gz'
            elif dataset_name == 'ontonotes':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/ontonotes_ner.tar.gz'
            else:
                raise RuntimeError('train.txt not found in "{}"'.format(data_path))
            data_path.mkdir(exist_ok=True, parents=True)
            download_decompress(url, data_path)

        dataset = {}

        files = [os.path.join(data_path, fname) for fname in ["train.txt", "valid.txt", "test.txt"]]

        for file_name in files:
            dataset_name = os.path.split(file_name)[1].split('.')[0]
            dataset[dataset_name] = self.parse_ner_file(file_name)

        return dataset

    def parse_ner_file(self, file_name: Path):
        samples = []
        word, tag = [], []
        expected_items = 2
        if self.provide_pos:
            pos = []
            expected_items += 1
        if self.provide_chunk:
            chunk = []
            expected_items += 1
        for line in open(file=file_name, mode="r", encoding='utf8').readlines():
            if line.startswith("-DOCSTART-"):
                continue

            items = line.strip().split()
            n = len(items)

            if n > 0:
                if n < expected_items:
                    raise ValueError(f"Missing value in line: {line}")

                word.append(items[0])
                tag.append(items[-1])
                if self.provide_pos:
                    pos.append(items[1])
                if self.provide_chunk:
                    chunk.append(items[2])

            elif n == 0:
                if len(word) == 0:
                    continue
                if not (self.provide_pos and self.provide_chunk):
                    x = word
                else:
                    x = (word,)
                    if self.provide_pos:
                        x += (pos,)
                    if self.provide_chunk:
                        x += (chunk,)

                if self.tagging_scheme == "iobes":
                    tag = self._iob_to_iobes(tag)

                samples.append((x, tag))
                word, tag = [], []
                if self.provide_pos:
                    pos = []
                if self.provide_chunk:
                    chunk = []
        return samples

    @staticmethod
    def _iob_to_iobes(tags):
        tag_map = {"BB": "S", "BO": "S", "IB": "E", "IO": "E"}
        tags = tags + ["O"]
        iobes_tags = []
        for i in range(len(tags) - 1):
            tagtag = tags[i][0] + tags[i + 1][0]
            if tagtag in tag_map:
                iobes_tags.append(tag_map[tagtag] + tags[i][1:])
            else:
                iobes_tags.append(tags[i])
        return iobes_tags
