from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import set_usr_dir

from deeppavlov.core.commands.train import train_model_from_config

CONFIG_PATH = 'config_en.json'
usr_dir = set_usr_dir(CONFIG_PATH)

###### Train all models
train_model_from_config(CONFIG_PATH)

##### Speak to a bot
interact_model(CONFIG_PATH)


