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
        prediction = self.model(observation)
        return prediction
