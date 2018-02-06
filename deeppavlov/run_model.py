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
from deeppavlov.core.commands.train import train_model_from_config, train_experimental
from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import set_usr_dir, get_usr_dir

# HCN
# skills/hcn/config.json

# HCN_new
# skills/go_bot/config.json

# Speller
# models/spellers/error_model/config_en.json
# models/spellers/error_model/config_ru.json
# models/spellers/error_model/config_ru_custom_vocab.json

# Intents classifier
# models/classifiers/intents/config_dstc2.json

# NER
# models/ner/slot_config.json

# PIPELINE_CONFIG_PATH = 'models/classifiers/intents/config_train.json'
PIPELINE_CONFIG_PATH = 'models/ner/ner_train.json'
# PIPELINE_CONFIG_PATH = 'models/ner/slot_config_train.json'
# PIPELINE_CONFIG_PATH = 'models/spellers/error_model/config_en.json'
# PIPELINE_CONFIG_PATH = 'models/spellers/error_model/config_ru_custom_vocab.json'
# PIPELINE_CONFIG_PATH = 'skills/go_bot/config.json'
set_usr_dir(PIPELINE_CONFIG_PATH)
try:
    # train_model_from_config(PIPELINE_CONFIG_PATH)
    train_experimental(PIPELINE_CONFIG_PATH)
    interact_model(PIPELINE_CONFIG_PATH)
# remove if usr_dir is empty:
finally:
    usr_dir = get_usr_dir()
    if not list(usr_dir.iterdir()):
        usr_dir.rmdir()
