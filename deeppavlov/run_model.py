from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import set_usr_dir

# HCN
# skills/hcn/config.json

# HCN_new
# skills/hcn_new/config.json

# Speller
# models/spellers/error_model/config_en.json

MODEL_CONFIG_PATH = 'skills/hcn_new/config.json'
set_usr_dir(MODEL_CONFIG_PATH)
train_model_from_config(MODEL_CONFIG_PATH)
interact_model(MODEL_CONFIG_PATH)
