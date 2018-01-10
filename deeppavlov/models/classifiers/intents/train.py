from deeppavlov.core.common.registry import REGISTRY
from deeppavlov.core.common.params import from_params
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model


def main(config_name='config.json'):

    train_model_from_config(config_path=config_name)

    # interact_model('config_infer.json')


if __name__ == '__main__':
    main()
