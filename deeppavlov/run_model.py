"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import set_deeppavlov_root


# PIPELINE_CONFIG_PATH = 'configs/intents/config_dstc2_train.json'
# PIPELINE_CONFIG_PATH = 'configs/intents/config_snips.json'
# PIPELINE_CONFIG_PATH = 'configs/ner/ner_dstc2_train.json'
# PIPELINE_CONFIG_PATH = 'configs/ner/ner_conll2003_train.json'
# PIPELINE_CONFIG_PATH = 'configs/ner/slot_config_train.json'
# PIPELINE_CONFIG_PATH = 'configs/error_model/config_en.json'
# PIPELINE_CONFIG_PATH = 'configs/error_model/config_ru.json'
PIPELINE_CONFIG_PATH = 'configs/go_bot/config_train.json'

set_deeppavlov_root(PIPELINE_CONFIG_PATH)

# train_model_from_config(PIPELINE_CONFIG_PATH)
train_model_from_config(PIPELINE_CONFIG_PATH)
interact_model(PIPELINE_CONFIG_PATH)
