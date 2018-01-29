from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register


@register('ner_dataset_reader')
class NerDatasetReader(DatasetReader):
    def read(self, file_path: str):
        dir_path = Path(file_path)
        files = list(dir_path.glob('*.txt'))
        assert any('train.txt' in str(file_path) for file_path in files)
        dataset = {}
        for file_name in files:
            file_name = str(file_name)
            name = (file_name.split('.')[0]).split('/')[-1]
            dataset[name] = self.parse_ner_file(file_name)
        return dataset

    @staticmethod
    def parse_ner_file(file_name: str):
        samples = []
        with open(file_name) as f:
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
