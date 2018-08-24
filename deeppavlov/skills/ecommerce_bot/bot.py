"""
Copyright 2018 Neural Networks and Deep Learning lab, MIPT
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

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import save_pickle, load_pickle
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.estimator import Estimator
from typing import List, Tuple
import spacy

log = get_logger(__name__)
nlp = spacy.load('en', parser=False)

@register("ecommerce_bot")
class EcommerceBot(Estimator):
    def __init__(self, save_path: str = None, **kwargs):
        self.save_path = save_path

    def fit(self, x) -> None:

        log.info('Items to nlp: '+str(len(x)))
        self.title_nlped = [nlp(item['Title']) for item in x]
        log.info('Titles are nlped')

        self.feat_nlped = [nlp(item['Title']+'.'+item['Feature']) if 'Feature' in item else nlp(item['Title']) for item in x]
        log.info('Features are nlped')
        
    def save(self, **kwargs) -> None:
        log.info("Saving model to {}".format(self.save_path))
        save_pickle((self.title_nlped, self.feat_nlped), expand_path(self.save_path))

    def load(self, **kwargs) -> None:
        pass
    
    def __call__(self, x, **kwargs):
        return [1,2]
