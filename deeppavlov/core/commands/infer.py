from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import REGISTRY

from deeppavlov.core.agent.agent import Agent
from deeppavlov.core.common.params import from_params


def build_model_from_config(config):
    model_config = config['model']
    model_config.update(config['vocabs'])
    model_name = model_config['name']
    model = from_params(REGISTRY[model_name], model_config)
    model.reset()
    return model


def build_agent_from_config(config_path: str):
    config = read_json(config_path)
    skill_configs = config['skills']
    commutator_config = config['commutator']
    return Agent(skill_configs, commutator_config)


def interact_agent(config_path):
    a = build_agent_from_config(config_path)
    commutator_name = a.commutator_config['name']
    commutator = from_params(REGISTRY[commutator_name], a.commutator_config)

    models = [build_model_from_config(sk) for sk in a.skill_configs]
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


def interact_model(config_path):
    config = read_json(config_path)
    model = build_model_from_config(config)

    while True:
        # get input from user
        context = input(':: ')

        # check for exit command
        if context == 'exit' or context == 'stop' or context == 'quit' or context == 'q':
            return

        try:
            pred = model.infer(context)
            print('>>', pred)
        except Exception as e:
            raise e
