from pathlib import Path
import json

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.registry import register


@register('squad_dataset_reader')
class SquadDatasetReader(DatasetReader):
    """
    Stanford Question Answering Dataset
    https://rajpurkar.github.io/SQuAD-explorer/
    """

    url = 'http://lnsigo.mipt.ru/export/datasets/squad-v1.1.tar.gz'

    def read(self, dir_path: str):
        dir_path = Path(dir_path)
        required_files = ['{}-v1.1.json'.format(dt) for dt in ['train', 'dev']]
        if not dir_path.exists():
            dir_path.mkdir()

        if not all((dir_path / f).exists() for f in required_files):
            download_decompress(self.url, dir_path)

        dataset = {}
        for f in required_files:
            data = json.load((dir_path / f).open('r'))
            if f == 'dev-v1.1.json':
                dataset['valid'] = data
            else:
                dataset['train'] = data

        return dataset
