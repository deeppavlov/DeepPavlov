from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress


@register('kbqa_reader')
class Conll2003DatasetReader(DatasetReader):
    """Class to read training datasets in CoNLL-2003 format"""

    def read(self, data_path: str):
        data_path = Path(data_path)
        files = list(data_path.glob('*.txt'))
        if 'test_set_with_answers.txt' not in {file_path.name for file_path in files}:
            url = 'http://files.deeppavlov.ai/kbqa/test_set_with_answers.txt'
            data_path.mkdir(exist_ok=True, parents=True)
            download_decompress(url, data_path)
            files = list(data_path.glob('*.txt'))
        dataset = {}

        dataset["test"] = self.parse_ner_file(files[0])
        return dataset

    def parse_ner_file(self, file_name: Path):
        samples = []
        with file_name.open(encoding='utf8') as f:
            for line in f:
                line_split = line.strip('\n').split('\t')
                samples.append((line_split[0], line_split[1]))

        return samples
