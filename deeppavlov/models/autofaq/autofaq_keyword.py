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
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
import pandas as pd
import numpy as np
import re

log = get_logger(__name__)


@register('autofaq_keyword')
class AutoFAQKeyword(Component):
    def __init__(self, faq_dataset_path, defined_phrases, *args, **kwargs):
        self.faq = pd.read_csv(faq_dataset_path)
        self.faq['question'] = [re.sub(r'[^\w\s]', '', q).lower() for q in self.faq['question']]
        self.defined_phrases = defined_phrases


    @overrides
    def __call__(self, question, *args, **kwargs):
        keywords = question[0].lower().split()
        if len(keywords) == 0:
            answer = ['Задайте вопрос, пожалуйста.']
        else:
            occurrence = np.zeros(len(self.faq))
            for i in range(len(keywords)):
                occurrence = occurrence + np.array([0 + (keywords[i] in q) for q in self.faq['question']])
            if sum(occurrence)==0:
                answer = ['К сожалению, я не знаю ответа на Ваш вопрос. Пожалуйста обратитесь к специалисту.']
            else:
                best_suited_id = np.argmax(occurrence)
                if sum(occurrence==occurrence[best_suited_id])>1:
                    answer = ['К сожалению, я не знаю ответа на Ваш вопрос. Пожалуйста обратитесь к специалисту.']
                else:
                    # answer = [self.faq.loc[best_suited_id,'question'], '\n', self.faq.loc[best_suited_id,'answer']]
                    answer = [self.faq.loc[best_suited_id,'answer']]

        return answer


