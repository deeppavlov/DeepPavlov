from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.commands.utils import set_vocab_path, build_agent_from_config, set_usr_dir
from deeppavlov.core.common.params import from_params
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.common import paths

from pathlib import Path


# TODO pass paths to local model configs to agent config.

def get_data(datareader_config, dataset_config, vocab_path):
    datareader_name = datareader_config['name']
    data_path = datareader_config.pop('data_path')

    data_reader = from_params(_REGISTRY[datareader_name], datareader_config)
    raw_data = data_reader.read(data_path)

    dataset_name = dataset_config['name']
    dataset_config['data'] = raw_data
    data = from_params(_REGISTRY[dataset_name], dataset_config)

    # data.save_vocab(data.iter_all('train'), vocab_path)
    return data


def train_agent_models(config_path: str):
    usr_dir = paths.USR_PATH
    a = build_agent_from_config(config_path)

    for skill_config in a.skill_configs:
        vocab_path = set_vocab_path()
        model_config = skill_config['model']
        model_name = model_config['name']

        if issubclass(_REGISTRY[model_name], Trainable):
            reader_config = skill_config['dataset_reader']
            reader = from_params(_REGISTRY[reader_config['name']], {})
            data = reader.read(reader_config.get('data_path', usr_dir))

            model = from_params(_REGISTRY[model_name], model_config, vocab_path=vocab_path)

            # TODO if has TFModel as attribute
            # TODO is model.train_now
            num_epochs = skill_config['num_epochs']
            num_tr_data = skill_config['num_train_instances']
            model.train(data, num_epochs, num_tr_data)
        else:
            print('Model {} is not an instance of Trainable, skip training.'.format(model_name))
            pass
            # raise NotImplementedError("This model is not an instance of TFModel class."
            #                           "Only TFModel instances can train for now.")


def train_model_from_config(config_path: str):
    usr_dir = paths.USR_PATH
    config = read_json(config_path)

    vocab_path = Path(usr_dir).joinpath('vocab.txt')

    data = get_data(config['dataset_reader'], config['dataset'], vocab_path)

    model_config = config['model']
    model_name = model_config['name']
    model = from_params(_REGISTRY[model_name], model_config, vocab_path=vocab_path)

    num_epochs = config['num_epochs']

    ####### Train
    model.train(data, num_epochs=num_epochs)
    # The result is a saved to user_dir trained model.
