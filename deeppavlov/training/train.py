from pathlib import Path

from deeppavlov.common import paths
from deeppavlov.common.file import read_json
from deeppavlov.common.registry import _REGISTRY
from deeppavlov.common.params import from_params


# TODO
def train_model(data, model):
    pass


def train_model_from_config(config_path: str, usr_dir_name='USR_DIR'):
    # make a serialization user dir
    root_ = Path(config_path).resolve().parent
    usr_dir_path = root_.joinpath(usr_dir_name)
    if not usr_dir_path.exists():
        usr_dir_path.mkdir()

    paths.USR_PATH = usr_dir_path

    config = read_json(config_path)

    dataset_config = config['dataset_reader']
    dataset_name = dataset_config['name']
    data_path = config['data_path']

    data_reader = from_params(_REGISTRY[dataset_name], dataset_config)
    data = data_reader.read(data_path)

    vocab_path = usr_dir_path.joinpath('vocab.txt')
    data_reader.save_vocab(data, vocab_path)

    model_config = config['model']
    model_name = model_config['name']
    model = from_params(_REGISTRY[model_name], model_config, vocab_path=vocab_path)

    num_epochs = config['num_epochs']
    num_tr_data = config['num_train_instances']

    ####### Train
    # TODO do batching in the train script. Now there are dialogs that somehow are batches.
    model.train(data, num_epochs, num_tr_data)

    # The result is a saved to user_dir trained model.
