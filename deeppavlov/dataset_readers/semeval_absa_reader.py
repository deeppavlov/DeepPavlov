from logging import getLogger
from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress

log = getLogger(__name__)


@register('semeval2015_absa_reader')
class SemEval2015DatasetReader(DatasetReader):
    """Class to read training datasets for ABSA task format"""

    def read(self,
             data_path: str,
             dataset_name: str = None,
            *args, **kwargs):
        data_path = Path(data_path)
        files = list(data_path.glob('*.txt'))
        if 'train.txt' not in {file_path.name for file_path in files}:
            if dataset_name == 'semeval2015':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/semeval2015_absa.tar.gz'
            else:
                raise RuntimeError('train.txt not found in "{}"'.format(data_path))
            data_path.mkdir(exist_ok=True, parents=True)
            download_decompress(url, data_path)
            files = list(data_path.glob('*.txt'))
        dataset = {}

        for file_name in files:
            name = file_name.with_suffix('').name
            dataset[name] = self.samples(file_name)
        return dataset

    def parse_file(self, file_name: Path):
        samples = []
        with file_name.open(encoding='utf8') as f:
            tokens = []
            pos_tags = []
            chunk_tags = []
            tags = []
            expected_items = 2 + int(self.provide_pos) + int(self.provide_chunk)
            for line in f:
                    items = line.split()
                    if len(items) < expected_items:
                        raise Exception(f"Input is not valid {line}")
                    tokens.append(items[0])
                    tags.append(items[-1])
            if tokens:
                x = tokens if not self.x_is_tuple else (tokens,)
                samples.append((x, tags))
                self.num_docs += 1
        return samples

