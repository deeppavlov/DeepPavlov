from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config

log = get_logger(__name__)


class Model:
    def __init__(self, server_config: dict):
        super(Model, self).__init__()

        self.server_config = server_config
        self.model_config = read_json(server_config['model_config_path'])
        self.model = build_model_from_config(self.model_config)

        self.in_x = self.model.in_x

    def infer(self, observation):
        if self.server_config['stateful']:
            # TODO: implemet stateful mode
            prediction = '"stateful" mode is not supported yet'
        elif self.server_config['use_history']:
            # TODO: implemet use-history mode
            prediction = '"use-history" mode is not supported yet'
        else:
            prediction = self.model(observation.content)

        return prediction
