import json

from deeppavlov.common.params import from_params
import deeppavlov.common.registry as registry
from deeppavlov.data.dataset import Dataset
from deeppavlov.data.dataset_readers.dataset_reader import DatasetReader
from deeppavlov.models.trainable import Trainable

from deeppavlov.data.dataset_readers.typos_kartaslov import TyposKartaslov
from deeppavlov.data.dataset_readers.typos_wikipedia import TyposWikipedia
from speller.typos_dataset import TyposDataset
from speller.models.error_model import ErrorModel


def main(config_name='config.json'):
    with open(config_name) as f:
        config = json.load(f)
    model_config = config['model']
    model: Trainable = from_params(registry.model(model_config['name']), model_config)

    reader_config = config['dataset_reader']
    reader: DatasetReader = registry.model(reader_config['name'])
    data = reader.read(reader_config.get('data_path', '../data'))

    dataset_config = config['dataset']
    dataset: Dataset = registry.model(dataset_config['name'])(data, **dataset_config)

    model.train(dataset.iter_all())
    model.save()


if __name__ == '__main__':
    main()
