from pathlib import Path

from deeppavlov.common import paths
from deeppavlov.common.file import read_json
from deeppavlov.common.registry import _REGISTRY
from deeppavlov.common.params import from_params
from deeppavlov.agent.agent import Agent
from deeppavlov.models.trainable import Trainable


# TODO do a separate training for agent and separate - for models.
# TODO pass paths to local model configs to agent config.

def set_usr_dir(config_path: str, usr_dir_name='USR_DIR'):
    # make a serialization user dir
    root_ = Path(config_path).resolve().parent
    usr_dir_path = root_.joinpath(usr_dir_name)
    if not usr_dir_path.exists():
        usr_dir_path.mkdir()
    paths.USR_PATH = usr_dir_path


def set_vocab_path():
    return paths.USR_PATH.joinpath('vocab.txt')


def build_agent_from_config(config_path: str):
    config = read_json(config_path)
    skill_configs = config['skills']
    commutator_config = config['commutator']
    return Agent(skill_configs, commutator_config)


def train_agent_models(config_path: str):
    set_usr_dir(config_path)
    a = build_agent_from_config(config_path)

    for skill_config in a.skill_configs:
        vocab_path = set_vocab_path()
        model_config = skill_config['model']
        model_name = model_config['name']

        if issubclass(_REGISTRY[model_name], Trainable):
            dataset_config = skill_config['dataset_reader']
            dataset_name = dataset_config['name']
            data_path = skill_config['data_path']

            data_reader = from_params(_REGISTRY[dataset_name], dataset_config)
            data = data_reader.read(data_path)
            data_reader.save_vocab(data, vocab_path)

            model = from_params(_REGISTRY[model_name], model_config, vocab_path=vocab_path)

            # TODO if has TFModel as attribute
            # TODO is model.train_now
            num_epochs = skill_config['num_epochs']
            num_tr_data = skill_config['num_train_instances']
            model.train(data, num_epochs, num_tr_data)
        else:
            pass
            # raise NotImplementedError("This model is not an instance of TFModel class."
            #                           "Only TFModel instances can train for now.")
