from deeppavlov.core.data.utils import download_decompress

from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register


@register('conll2003_reader')
class Conll2003DatasetReader(DatasetReader):
    """Class to read training datasets in CONLL2003 format"""
    def download_conll(self, dir_path):
        download_decompress('http://files.deeppavlov.ai/deeppavlov_data/conll2003_v2.tar.gz', dir_path)

    def read(self, dir_path: str, dataset_name='conll2003', provide_pos=False):
        self.provide_pos = provide_pos
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
                        token, *_, pos, tag = line.split()
                        pos_tags.append(pos)
                    else:
                        token, *_, tag = line.split()
                    tags.append(tag)
                    tokens.append(token)

        return samples
