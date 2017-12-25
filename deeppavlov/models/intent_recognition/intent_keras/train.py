from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.common.params import from_params
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model

import json


def main(config_name='intent_config.json'):


    with open(config_name) as f:
        config = json.load(f)

    train_model_from_config(config_path=config_name)


    # interact_model('intent_config_infer.json')


if __name__ == '__main__':
    main()
