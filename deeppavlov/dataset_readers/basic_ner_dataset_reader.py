from deeppavlov.core.data.utils import download_decompress
import sys

from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register


@register('ner_dataset_reader')
class NerDatasetReader(DatasetReader):

    def download_conll(self, dir_path):
        download_decompress('http://lnsigo.mipt.ru/export/datasets/conll2003.tar.gz', dir_path)

    def read(self, dir_path: str, dataset_name='conll2003'):
        dir_path = Path(dir_path)
        files = list(dir_path.glob('*.txt'))
        if 'train.txt' not in {file_path.name for file_path in files}:
            if dataset_name == 'conll2003':
                dir_path.mkdir(exist_ok=True, parents=True)
                self.download_conll(dir_path)
                files = list(dir_path.glob('*.txt'))
            else:
                raise RuntimeError('train.txt not found in "{}"'.format(dir_path))
        dataset = {}
        for file_name in files:
            name = file_name.with_suffix('').name
            dataset[name] = self.parse_ner_file(file_name)
        return dataset

    @staticmethod
    def parse_ner_file(file_name: Path):
        samples = []
        with file_name.open() as f:
            tokens = []
            tags = []
            for line in f:
                # Check end of the document
                if len(line) < 2 or 'DOCSTART' in line:
                    if len(tokens) > 0:
                        samples.append((tokens, tags, ))
                        tokens = []
                        tags = []
                else:
                    token, *_, tag = line.split()
                    tokens.append(token)
                    tags.append(tag)
        return samples
