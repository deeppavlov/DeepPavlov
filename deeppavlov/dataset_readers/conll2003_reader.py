from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress


@register('conll2003_reader')
class Conll2003DatasetReader(DatasetReader):
    """Class to read training datasets in CoNLL-2003 format"""

    def read(self, data_path: str, dataset_name=None, provide_pos=False):
        self.provide_pos = provide_pos
        data_path = Path(data_path)
        files = list(data_path.glob('*.txt'))
        if 'train.txt' not in {file_path.name for file_path in files}:
            if dataset_name == 'conll2003':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/conll2003_v2.tar.gz'
            elif dataset_name == 'collection_rus':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/collection5.tar.gz'
            elif dataset_name == 'ontonotes':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/ontonotes_ner.tar.gz'
            else:
                raise RuntimeError('train.txt not found in "{}"'.format(data_path))
            data_path.mkdir(exist_ok=True, parents=True)
            download_decompress(url, data_path)
            files = list(data_path.glob('*.txt'))
        dataset = {}

        for file_name in files:
            name = file_name.with_suffix('').name
            dataset[name] = self.parse_ner_file(file_name)
        return dataset

    def parse_ner_file(self, file_name: Path):
        samples = []
        with file_name.open(encoding='utf8') as f:
            tokens = ['<DOCSTART>']
            pos_tags = ['O']
            tags = ['O']
            for line in f:
                # Check end of the document
                if 'DOCSTART' in line:
                    if len(tokens) > 1:
                        if self.provide_pos:
                            samples.append(((tokens, pos_tags), tags, ))
                        else:
                            samples.append((tokens, tags,))
                        tokens = []
                        pos_tags = []
                        tags = []
                elif len(line) < 2:
                    if len(tokens) > 0:
                        if self.provide_pos:
                            samples.append(((tokens, pos_tags), tags, ))
                        else:
                            samples.append((tokens, tags,))
                        tokens = []
                        pos_tags = []
                        tags = []
                else:
                    if self.provide_pos:
                        token, pos, *_, tag = line.split()
                        pos_tags.append(pos)
                    else:
                        token, *_, tag = line.split()
                    tags.append(tag)
                    tokens.append(token)

        return samples
