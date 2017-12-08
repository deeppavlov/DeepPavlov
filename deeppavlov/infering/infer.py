from pathlib import Path

from deeppavlov.common import paths
from deeppavlov.common.file import read_json
from deeppavlov.common.params import from_params
from deeppavlov.common.registry import _REGISTRY
from deeppavlov.agent.agent import Agent


def get_usr_dir(config_path: str, usr_dir_name='USR_DIR'):
    # make a serialization user dir
    root_ = Path(config_path).resolve().parent
    usr_dir_path = root_.joinpath(usr_dir_name)
    if not usr_dir_path.exists():
        usr_dir_path.mkdir()
    paths.USR_PATH = usr_dir_path
    return usr_dir_path


def set_vocab_path():
    return paths.USR_PATH.joinpath('vocab.txt')


def build_agent_from_config(config_path: str):
    config = read_json(config_path)
    skill_configs = config['skills']
    commutator_config = config['commutator']
    return Agent(skill_configs, commutator_config)


def get_agent_models(config_path: str, a: Agent):
    get_usr_dir(config_path)
    models = []
    for skill_config in a.skill_configs:
        vocab_path = set_vocab_path()
        model_config = skill_config['model']
        model_name = model_config['name']

        model = from_params(_REGISTRY[model_name], model_config, vocab_path=vocab_path)
        model.reset()
        models.append(model)
    return models


def interact_agent(config_path):
    a = build_agent_from_config(config_path)
    commutator_name = a.commutator_config['name']
    commutator = from_params(_REGISTRY[commutator_name], commutator_name)

    models = get_agent_models(config_path, a)
    while True:
        # get input from user
        context = input(':: ')

        # check for exit command
        if context == 'exit' or context == 'stop' or context == 'quit' or context == 'q':
            return

        predictions = []
        for model in models:
            predictions.append(model.infer(context))
        pred = commutator.infer(predictions, a.history)
        print('>>', pred)

        a.history.append({'context': context, "predictions": predictions, "winner": pred})


def get_model_from_config(config_path, usr_dir_name='USR_DIR'):
    config = read_json(config_path)

    # make a serialization user dir
    root_ = Path(config_path).resolve().parent
    usr_dir_path = root_.joinpath(usr_dir_name)

    paths.USR_PATH = usr_dir_path

    vocab_path = Path(usr_dir_path).joinpath('vocab.txt')

    model_config = config['model']
    model_name = model_config['name']
    model = from_params(_REGISTRY[model_name], model_config, vocab_path=vocab_path)
    return model


def interact_model(config_path):
    model = get_model_from_config(config_path)
    model.reset()
    # while True:
    #     model.interact()
    while True:
        # get input from user
        context = input(':: ')

        # check for exit command
        if context == 'exit' or context == 'stop' or context == 'quit' or context == 'q':
            return

        pred = model.infer(context)
        print('>>', pred)