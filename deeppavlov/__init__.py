# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

try:
    from .configs import configs
    # noinspection PyUnresolvedReferences
    from .core.commands.infer import build_model
    # noinspection PyUnresolvedReferences
    from .core.commands.train import train_evaluate_model_from_config
    from .download import deep_download

    # TODO: make better and add typing
    def train_model(config, download=False):
        train_evaluate_model_from_config(config, download=download)
        return build_model(config, load_trained=True)
except ImportError:
    'Assuming that requirements are not yet installed'

__version__ = '0.0.9'
__author__ = 'Neural Networks and Deep Learning lab, MIPT'
__description__ = 'An open source library for building end-to-end dialog systems and training chatbots.'
__keywords__ = ['NLP', 'NER', 'SQUAD', 'Intents', 'Chatbot']
__license__ = 'Apache License, Version 2.0'
__email__ = 'info@ipavlov.ai'

# check version
assert sys.hexversion >= 0x3060000, 'Does not work in python3.5 or lower'
