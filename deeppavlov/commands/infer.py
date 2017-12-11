from deeppavlov.common.file import read_json
from deeppavlov.common.params import from_params
from deeppavlov.common.registry import _REGISTRY
from deeppavlov.agent.agent import Agent

from .utils import set_usr_dir, set_vocab_path, build_agent_from_config, USR_DIR


def build_agent_models_from_config(config_path: str, a: Agent, usr_dir_name=USR_DIR):
    set_usr_dir(config_path, usr_dir_name)
    # TODO get rid of the collision when different models save vocab to the same path
    vocab_path = set_vocab_path()
    models = []
    for skill_config in a.skill_configs:
        model_config = skill_config['model']
        model_name = model_config['name']

        model = from_params(_REGISTRY[model_name], model_config, vocab_path=vocab_path)
        models.append(model)
    return models


def interact_agent(config_path):
    a = build_agent_from_config(config_path)
    commutator_name = a.commutator_config['name']
    commutator = from_params(_REGISTRY[commutator_name], commutator_name)

    models = build_agent_models_from_config(config_path, a)
    while True:
        # get input from user
        context = input(':: ')

        # check for exit command
        if context == 'exit' or context == 'stop' or context == 'quit' or context == 'q':
            return

        predictions = []
        for model in models:
            predictions.append({model.__class__.__name__: model.infer(context)})
        idx, name, pred = commutator.infer(predictions, a.history)
        print('>>', pred)

        a.history.append({'context': context, "predictions": predictions,
                          "winner": {"idx": idx, "model": name, "prediction": pred}})
        print("Current history: {}".format(a.history))


def build_model_from_config(config_path, usr_dir_name=USR_DIR):
    set_usr_dir(config_path, usr_dir_name)
    vocab_path = set_vocab_path()
    model = _get_model(config_path, vocab_path)
    return model


def _get_model(config_path, vocab_path):
    config = read_json(config_path)
    model_config = config['model']
    model_name = model_config['name']
    model = from_params(_REGISTRY[model_name], model_config, vocab_path=vocab_path)
    return model


def interact_model(config_path):
    model = build_model_from_config(config_path)
    model.reset()
    while True:
        # get input from user
        context = input(':: ')

        # check for exit command
        if context == 'exit' or context == 'stop' or context == 'quit' or context == 'q':
            return

        pred = model.infer(context)
        print('>>', pred)
