import json
from os import path, mkdir

from deeppavlov.common.registry import _REGISTRY
from deeppavlov.common.params import from_params


# def make_usr_dir(config_path):
#     dir_path = path.dirname(path.abspath(config_path))
#     dir_name = 'USR_DIR'
#     rUSR_DIR = path.join(dir_path, dir_name)
#     if not path.exists(USR_DIR):
#         mkdir(USR_DIR)


def read_config(config_path):
    with open(config_path) as f:
        return json.load(f)


# TODO
def train_model(data, model):
    pass


def train_model_from_config(config_path, ser_dir):


    config = read_config(config_path)

    dataset_config = config['dataset_reader']
    dataset_name = dataset_config['name']
    data_path = config['data_path']

    data_reader = from_params(_REGISTRY[dataset_name], dataset_config)
    data = data_reader.read(data_path)

    vocab_path = path.join(ser_dir, 'vocab.txt')
    data_reader.save_vocab(data, vocab_path)

    model_config = config['model']
    model_name = model_config['name']
    model = from_params(_REGISTRY[model_name], model_config, vocab_path=vocab_path)

    num_epochs = config['num_epochs']
    num_tr_data = config['num_train_instances']

    ####### Train
    # TODO do batching in the train script
    model.train(data, num_epochs, num_tr_data)

    # The result should be a saved trained model.
