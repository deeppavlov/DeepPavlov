from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.utils import set_usr_dir

##### Train and speak to HCN_go skill separately
MODEL_CONFIG_PATH = 'config.json'
usr_dir = set_usr_dir(MODEL_CONFIG_PATH)
train_model_from_config(MODEL_CONFIG_PATH)
interact_model(MODEL_CONFIG_PATH)



