from deeppavlov.infering.infer import interact

from speller.models.error_model import ErrorModel

CONFIG_PATH = 'config_en.json'
###### Train all models
# train_model_from_config(CONFIG_PATH)

##### Speak to a bot
interact(CONFIG_PATH)


