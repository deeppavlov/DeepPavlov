import deeppavlov.core.common.registry as registry
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.registry import _REGISTRY


def train(config_path, usr_dir):
    config = read_json(config_path)
    model_config = config['model']
    model_name = model_config['name']

    # Path for models should be specified here:
    model = from_params(_REGISTRY[model_name], model_config, models_path=usr_dir)

    reader_config = config['dataset_reader']
    reader = _REGISTRY[reader_config['name']]
    data = reader.read(reader_config.get('data_path', usr_dir))

    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    dataset = from_params(_REGISTRY[dataset_name], dataset_config, data=data)

    model.train(dataset.iter_all())
    model.save()


# if __name__ == '__main__':
#     train('config_en.json', )
